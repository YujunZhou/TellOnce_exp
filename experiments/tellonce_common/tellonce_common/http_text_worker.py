"""One bounded HTTP attempt in an empty environment; keys arrive via stdin."""
import http.client
import json
import math
import os
from pathlib import Path
import re
import ssl
import sys
import time
from email.utils import parsedate_to_datetime

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tellonce_common.http_text_contract import HTTPTextConfig, SYNTHETIC_KEY, GEMMA_MODEL, clean_usage, openai_usage, endpoint, response_text, strict_json
from tellonce_common.process_control import bind_parent_lifetime


class RepeatedInvalidJSON(ValueError):
    """A repeated bare-word prefix can never become a valid JSON selection."""


def repeated_invalid_json(text):
    # Never reject repetition inside JSON strings, arrays, objects or fences.
    # Require a long all-word prefix: later output cannot repair this prefix.
    if len(text) < 512 or not re.fullmatch(r'[A-Za-z\s]+', text):
        return False
    words = text.split()
    return len(words) >= 64 and len(set(words)) == 1


def read_selection_stream(response, config, key):
    """Assemble one bounded SSE completion; preserve normal final validation."""
    value = {'choices': [{'message': {'role': None, 'content': ''}, 'finish_reason': None}]}
    choice = value['choices'][0]
    total = 0
    finished = False
    usage_trailer = False
    fields = []
    while True:
        line = response.readline(4*1024*1024+1)
        total += len(line)
        if total > 4*1024*1024:
            raise ValueError('response_budget')
        if not line:
            raise ValueError('stream ended without completion marker')
        line = line.decode('utf-8').rstrip('\r\n')
        if line.startswith(':'):
            continue
        if line:
            if line.startswith('data:'):
                fields.append(line[5:].lstrip(' '))
            elif not line.startswith(('event:', 'id:', 'retry:')):
                raise ValueError('unexpected SSE field')
            continue
        if not fields:
            continue
        data = '\n'.join(fields)
        fields = []
        if key in data:
            raise ValueError('credential_in_response')
        if data == '[DONE]':
            if not finished:
                raise ValueError('stream has no final choice')
            return value
        chunk = strict_json(data)
        if not isinstance(chunk, dict) or chunk.get('error'):
            raise ValueError('invalid stream chunk')
        for name in ('id', 'model', 'provider'):
            if chunk.get(name) is not None:
                if name in value and value[name] != chunk[name]:
                    raise ValueError('stream response identity changed')
                value[name] = chunk[name]
        if value.get('model', config.model) != config.model:
            raise ValueError('stream model changed')
        if config.provider == 'openrouter' and 'provider' in value and value['provider'] not in config.serving_providers:
            raise ValueError('stream provider changed')
        if chunk.get('usage') is not None:
            value['usage'] = clean_usage(chunk['usage'])
        choices = chunk.get('choices')
        if choices == [] and chunk.get('usage') is not None:
            continue
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError('stream must have one choice')
        part = choices[0]
        if type(part.get('index')) is not int or part['index'] != 0:
            raise ValueError('invalid stream choice order')
        delta = part.get('delta')
        if not isinstance(delta, dict) or delta.get('role', 'assistant') != 'assistant' or delta.get('tool_calls') or delta.get('function_call'):
            raise ValueError('stream is not assistant text')
        if finished:
            # OpenRouter repeats the finished, empty choice on its usage
            # trailer. This is metadata, never an additional completion.
            if (usage_trailer or chunk.get('usage') is None or
                    part.get('finish_reason') != choice['finish_reason'] or
                    set(delta)-{'role','content'} or delta.get('content') not in (None,'')):
                raise ValueError('invalid stream choice order')
            usage_trailer = True
            continue
        if 'role' in delta:
            choice['message']['role'] = delta['role']
        text = delta.get('content')
        if text is not None:
            if not isinstance(text, str):
                raise ValueError('stream content is not text')
            choice['message']['content'] += text
            if key in choice['message']['content']:
                raise ValueError('credential_in_response')
            if repeated_invalid_json(choice['message']['content']):
                raise RepeatedInvalidJSON('invalid repeated selection prefix')
        if part.get('finish_reason') is not None:
            choice['finish_reason'] = part['finish_reason']
            finished = True


def retry_after(headers):
    try:
        if headers.get('retry-after-ms') is not None:
            value=float(headers['retry-after-ms'])/1000
        else:
            raw=headers.get('Retry-After')
            if raw is None:
                return None
            try:
                value=float(raw)
            except ValueError:
                value=max(0,parsedate_to_datetime(raw).timestamp()-time.time())
        return value if math.isfinite(value) and value>=0 else None
    except (ValueError,TypeError,OverflowError):
        return None


def execute(request,metrics):
    config=HTTPTextConfig(**request['config'])
    if type(request.get('synthetic')) is not bool:
        raise ValueError('explicit HTTP execution mode required')
    url=endpoint(request['endpoint'],synthetic=request['synthetic'],provider=config.provider)
    key=request['api_key']
    if (not isinstance(key,str) or not 1<=len(key)<=4096 or not key.isascii() or
            any(ord(c)<=32 or ord(c)>=127 for c in key) or
            (request['synthetic'] and key!=SYNTHETIC_KEY)):
        raise ValueError('invalid explicit HTTP text credential')
    if not 0<request['timeout_s']<=300:
        raise ValueError('bounded HTTP socket timeout required')
    payload=config.payload(request['messages'])
    stream_selection=request.get('stream_json_selection',False)
    if type(stream_selection) is not bool or (stream_selection and
            not (config.provider=='openrouter' and config.model==GEMMA_MODEL)):
        raise ValueError('stream selection requires the admitted Gemma route')
    if stream_selection:
        payload.update(stream=True,stream_options={'include_usage':True})
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
    connection=(http.client.HTTPConnection(url.hostname,url.port,timeout=request['timeout_s']) if request['synthetic'] else
                http.client.HTTPSConnection(url.hostname,url.port,timeout=request['timeout_s'],context=ssl.create_default_context()))
    try:
        metrics['http_attempts_started']=1
        # http.client neither follows redirects nor inherits HTTP proxy vars.
        # A new connection is used once, with no implicit request retry.
        connection.request('POST',url.path,body=data,headers={
            'Authorization':'Bearer '+key,'Content-Type':'application/json',
            'Accept':'text/event-stream' if stream_selection else 'application/json'})
        response=connection.getresponse()
        metrics['http_status']=response.status
        metrics['retry_after_s']=retry_after(response.headers)
        if response.status!=200:
            if config.provider=='openai':
                # Read only bounded error metadata; never publish the body or message.
                try:
                    raw=response.read(65537)
                    error=strict_json(raw).get('error',{}) if len(raw)<=65536 else {}
                    code=error.get('code')
                    if code in {'insufficient_quota','billing_hard_limit_reached'}:
                        metrics['failure_code']='provider_credit_exhausted'
                    elif response.status in {401,403}:
                        metrics['failure_code']='provider_authorization_failed'
                    elif response.status==429:
                        metrics['failure_code']='provider_rate_limited'
                    elif code=='context_length_exceeded':
                        metrics['failure_code']='provider_context_exceeded'
                except (ValueError,TypeError,AttributeError):
                    pass
            raise RuntimeError('http_status')  # Never forward the provider error body.
        if stream_selection and response.headers.get_content_type()=='text/event-stream':
            value=read_selection_stream(response,config,key)
        else:
            raw=response.read(4*1024*1024+1)
            if len(raw)>4*1024*1024:
                raise RuntimeError('response_budget')
            value=strict_json(raw)
        pending=[value]
        while pending:
            item=pending.pop()
            if isinstance(item,dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item,list):
                pending.extend(item)
            elif isinstance(item,str) and key in item:
                raise RuntimeError('credential_in_response')
        if not isinstance(value,dict):
            raise ValueError('provider response is not an object')
        metrics['usage']=(openai_usage(value.get('usage')) if config.provider=='openai'
                          else clean_usage(value.get('usage')))
        if config.provider=='openai':
            output=value.get('output')
            messages=output if isinstance(output,list) else []
            if any(isinstance(part,dict) and part.get('type')=='refusal'
                    for item in messages if isinstance(item,dict) and isinstance(item.get('content'),list)
                    for part in item['content']):
                metrics['failure_code']='model_refusal'
                raise ValueError('model_refusal')
            text=response_text(value,config)
            metrics.update(provider='OpenAI',finish_reason='stop',system_fingerprint=None)
            return {'text':text,'response_id':value['id'],'model':value['model']}
        if config.provider=='openrouter':
            serving=value.get('provider')
            metrics['provider']=serving if isinstance(serving,str) and 1<=len(serving)<=256 else None
        choices=value.get('choices')
        finish=(choices[0].get('finish_reason') if isinstance(choices,list) and len(choices)==1
                and isinstance(choices[0],dict) else None)
        metrics['finish_reason']=finish if finish in {'stop','length','tool_calls','content_filter','function_call'} else None
        # The parent retains identity-checked partial output as private failure
        # evidence, then rejects finish_reason=length before any successful
        # receipt or validator. Do not discard that evidence in this worker.
        text=response_text(value,config,allow_truncated=True)
        fingerprint=value.get('system_fingerprint')
        metrics['system_fingerprint']=fingerprint if isinstance(fingerprint,str) and len(fingerprint)<=256 else None
        return {'text':text,'response_id':value['id'],'model':value['model']}
    finally:
        connection.close()


def main():
    metrics={'http_attempts_started':0,'usage':None}
    started=time.monotonic()
    result={}
    try:
        if len(sys.argv)!=3 or sys.argv[1]!='--owner-pid':
            raise ValueError('explicit parent lifetime required')
        bind_parent_lifetime(int(sys.argv[2]))
        # A 512 KiB text body can expand sixfold when JSON escapes controls.
        raw=sys.stdin.buffer.read(4*1024*1024+1)
        if len(raw)>4*1024*1024:
            raise ValueError('HTTP worker input exceeds its budget')
        request=strict_json(raw)
        result=execute(request,metrics)
        result['status']='completed'
    except Exception as exc:
        result={'status':'failed','error_type':type(exc).__name__}
    metrics['worker_wall_s']=time.monotonic()-started
    print(json.dumps({'result':result,'metrics':metrics},ensure_ascii=False,allow_nan=False),flush=True)
    return 0 if result.get('status')=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
