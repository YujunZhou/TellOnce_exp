"""Explicit provider-bound text requests with bounded same-model routing."""
from dataclasses import dataclass
import json
import math
import re
from urllib.parse import urlsplit
from .text_input_budget import MAX_TEXT_INPUT_BYTES, check_text_input, validate_input_limit


DEEPINFRA_ENDPOINT='https://api.deepinfra.com/v1/openai/chat/completions'
OPENROUTER_ENDPOINT='https://openrouter.ai/api/v1/chat/completions'
OPENAI_ENDPOINT='https://api.openai.com/v1/responses'
LUNA_API_MODEL='gpt-6-luna'
DEEPSEEK_MODEL='deepseek/deepseek-v4.1-flash'
GEMMA_MODEL='google/gemma-4-31b-it'
PROMPT_CACHE_PREFIX_BYTES=2048


def prompt_cache_key(messages):
    """Stable key for requests whose conversation starts with the same instruction prefix.

    Prompt caching matches on the token prefix; the key only steers requests with one prefix
    to one cache, so it is derived from the first bytes of the conversation (role-tagged) and
    never from anything that varies per call after that prefix."""
    import hashlib
    head=''.join(f"{message['role']}\x1f{message['content']}\x1e" for message in messages).encode('utf-8')[:PROMPT_CACHE_PREFIX_BYTES]
    return 'tellonce-'+hashlib.sha256(head).hexdigest()[:32]
OPENROUTER_MODELS={DEEPSEEK_MODEL:('deepseek','DeepSeek',1048576),
                   GEMMA_MODEL:('deepinfra/turbo','DeepInfra',262144),
                   'google/gemini-3.8-flash':('google-ai-studio','Google AI Studio',1048576)}
ENDPOINTS={'deepinfra':DEEPINFRA_ENDPOINT,'openrouter':OPENROUTER_ENDPOINT,'openai':OPENAI_ENDPOINT}
SYNTHETIC_KEY='SYNTHETIC_HTTP_TEXT_KEY_NOT_A_CREDENTIAL'


@dataclass(frozen=True)
class HTTPTextConfig:
    model: str
    max_tokens: int = 16384
    temperature: float = 0.0
    reasoning_effort: str | None = None
    input_max_bytes: int = MAX_TEXT_INPUT_BYTES
    provider: str = 'deepinfra'
    openrouter_upstream: str | None = None

    def __post_init__(self):
        validate_input_limit(self.input_max_bytes)
        if (not isinstance(self.provider,str) or self.provider not in ENDPOINTS or
                self.provider=='openrouter' and self.model not in OPENROUTER_MODELS):
            raise ValueError('explicit admitted HTTP provider and model required')
        if (not isinstance(self.model,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9/_.:-]{0,255}',self.model) or
                'sonnet' in self.model.lower() or type(self.max_tokens) is not int or not 1<=self.max_tokens<=16384 or
                type(self.temperature) not in {int,float} or not math.isfinite(self.temperature) or
                not 0<=self.temperature<=2 or self.reasoning_effort not in {None,'none','low','medium','high'}):
            raise ValueError('explicit supported text model and bounded generation settings required')
        if self.provider=='deepinfra' and self.model==GEMMA_MODEL:
            raise ValueError('use the official case-sensitive identifier google/gemma-4-31B-it')
        if self.provider=='openai' and (self.model!=LUNA_API_MODEL or self.reasoning_effort not in {'none','low','medium'}):
            raise ValueError('official API text roles require gpt-6-luna with explicit none, low or medium reasoning')
        if self.openrouter_upstream is not None and not (
                self.provider=='openrouter' and self.model==DEEPSEEK_MODEL and
                self.openrouter_upstream in ('deepinfra/fp8','novita/fp8','alibaba','low-cost-auto')):
            raise ValueError('the explicit OpenRouter upstream is not admitted for this model')

    @property
    def serving_route(self):
        if self.provider!='openrouter':
            raise ValueError('serving route is specific to OpenRouter')
        if self.openrouter_upstream=='low-cost-auto':
            raise ValueError('automatic routing has multiple admitted serving providers')
        if self.openrouter_upstream=='deepinfra/fp8':
            return ('deepinfra/fp8','DeepInfra')
        if self.openrouter_upstream=='novita/fp8':
            return ('novita/fp8','Novita')
        if self.openrouter_upstream=='alibaba':
            return ('alibaba','Alibaba')
        return OPENROUTER_MODELS[self.model][:2]

    @property
    def serving_providers(self):
        if self.provider!='openrouter':
            raise ValueError('serving providers are specific to OpenRouter')
        if self.openrouter_upstream=='low-cost-auto':
            return ('DeepSeek','Alibaba','DeepInfra')
        return (self.serving_route[1],)

    def payload(self,messages):
        if (not isinstance(messages,list) or not messages or len(messages)>128 or
                any(not isinstance(message,dict) or set(message)!={'role','content'} or
                    message['role'] not in {'system','user','assistant'} or not isinstance(message['content'],str)
                    for message in messages) or messages[-1]['role']!='user'):
            raise ValueError('explicit text conversation ending in the current user request required')
        check_text_input(*(message['content'] for message in messages), limit=self.input_max_bytes)
        if self.provider=='openai':
            # Reasoning consumes output tokens. Never send unsupported sampling
            # controls, tools, or a server-side conversation/storage request.
            # The cache key routes requests that share an instruction prefix to
            # the same cache; it changes nothing about the model input or output.
            return {'model':self.model,'input':messages,'reasoning':{'effort':self.reasoning_effort},
                    'max_output_tokens':self.max_tokens,'stream':False,'store':False,'tools':[],
                    # 2026-10-05: no extended (24 h) cache retention. Measured hit rate of the check prompts was
                    # 0.9-2.4 % of prompt tokens over four runs, so retention only adds cache writes; the
                    # key still groups a shared prefix within the provider's default window.
                    'prompt_cache_key':prompt_cache_key(messages)}
        result={'model':self.model,'messages':messages,'temperature':self.temperature,
                'max_tokens':self.max_tokens,'stream':False,'n':1}
        if self.reasoning_effort is not None:
            result['reasoning_effort']=self.reasoning_effort
        if self.provider=='openrouter':
            # Preserve the original input and model; only an explicitly
            # selected route may fail over within this bounded provider set.
            result.pop('n')
            if self.openrouter_upstream=='low-cost-auto':
                result['provider']={'only':['deepseek','alibaba','deepinfra/fp8'],
                    'allow_fallbacks':True,'require_parameters':True,
                    'max_price':{'prompt':.2,'completion':.6}}
            else:
                slug=self.serving_route[0]
                result['provider']={'only':[slug],'allow_fallbacks':False,'require_parameters':True}
                if slug=='google-ai-studio':
                    result['provider']['ignore']=['google-ai-studio/flex','google-ai-studio/priority']
                if self.model==GEMMA_MODEL:
                    result['provider']['max_price']={'prompt':.2,'completion':.6}
            result['transforms']=[]
            if self.reasoning_effort is not None:
                result.pop('reasoning_effort')
                result['reasoning']=({'enabled':False} if self.reasoning_effort=='none'
                                     else {'effort':self.reasoning_effort})
        return result


def endpoint(value, *, synthetic, provider='deepinfra'):
    if not isinstance(provider,str) or provider not in ENDPOINTS:
        raise ValueError('unknown HTTP text provider')
    if not synthetic:
        if value!=ENDPOINTS[provider]:
            raise ValueError('live text credentials are bound to the selected official provider endpoint')
    else:
        url=urlsplit(value)
        if (url.scheme!='http' or url.hostname!='127.0.0.1' or url.port is None or url.username or
                url.password or url.query or url.fragment or url.path!='/v1/openai/chat/completions'):
            raise ValueError('synthetic text requests must stay on the local fixture endpoint')
    return urlsplit(value)


def unwrap_json_fence(text):
    """Remove only a complete fence containing valid JSON; never extract prose."""
    wrapped=re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*',text,re.DOTALL)
    if wrapped:
        try:
            strict_json(wrapped[1])
        except (ValueError, TypeError):
            return text
        return wrapped[1]
    return text


def strict_json(data):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:
                raise ValueError('duplicate provider JSON field')
            value[key]=item
        return value
    return json.loads(data,object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite provider JSON field')))


def response_text(value,config,*,allow_truncated=False):
    if config.provider=='openai':
        return openai_response_text(value,config)
    if not isinstance(value,dict) or value.get('model')!=config.model:
        raise ValueError('provider changed the declared model')
    if config.provider=='openrouter' and value.get('provider') not in config.serving_providers:
        raise ValueError('OpenRouter changed the declared serving provider')
    choices=value.get('choices')
    if not isinstance(choices,list) or len(choices)!=1:
        raise ValueError('provider must return exactly one completion')
    choice=choices[0]
    if not isinstance(choice,dict) or not isinstance(choice.get('message'),dict):
        raise ValueError('provider choice must contain an assistant message object')
    message=choice.get('message',{})
    allowed_finish={'stop','length'} if allow_truncated is True else {'stop'}
    if (choice.get('finish_reason') not in allowed_finish or message.get('role')!='assistant' or
            message.get('tool_calls') or message.get('function_call') or
            not isinstance(message.get('content'),str) or not message['content'].strip()):
        raise ValueError('provider returned an incomplete, empty or tool-using response')
    if not isinstance(value.get('id'),str) or not 1<=len(value['id'])<=512:
        raise ValueError('provider response identity is missing')
    return message['content']


def openai_response_text(value,config):
    if (not isinstance(value,dict) or value.get('model')!=config.model or
            value.get('status')!='completed' or value.get('error') or
            value.get('incomplete_details') or not isinstance(value.get('reasoning'),dict) or
            value['reasoning'].get('effort')!=config.reasoning_effort or
            not isinstance(value.get('id'),str) or not 1<=len(value['id'])<=512):
        raise ValueError('official response is incomplete or changed model/reasoning')
    output=value.get('output')
    if not isinstance(output,list) or not output:
        raise ValueError('official response has no output')
    parts=[]
    for item in output:
        if not isinstance(item,dict):
            raise ValueError('invalid official output item')
        if item.get('type')=='reasoning':
            continue
        if (item.get('type')!='message' or item.get('role')!='assistant' or
                item.get('status')!='completed' or not isinstance(item.get('content'),list)):
            raise ValueError('official response contains unexpected tools or incomplete text')
        for part in item['content']:
            if not isinstance(part,dict) or part.get('type')!='output_text' or not isinstance(part.get('text'),str):
                raise ValueError('official response contains refusal or non-text output')
            parts.append(part['text'])
    text='\n'.join(parts)
    if not text.strip():
        raise ValueError('official response has no assistant text')
    return text


def openai_usage(value):
    if not isinstance(value,dict):
        raise ValueError('official response usage is missing')
    return clean_usage({'prompt_tokens':value.get('input_tokens'),
        'completion_tokens':value.get('output_tokens'),'total_tokens':value.get('total_tokens'),
        'prompt_tokens_details':value.get('input_tokens_details'),
        'completion_tokens_details':value.get('output_tokens_details')})


def clean_usage(usage):
    if usage is None:
        return None
    if (not isinstance(usage,dict) or any(type(usage.get(key)) is not int or usage[key]<0
            for key in ('prompt_tokens','completion_tokens'))):
        raise ValueError('provider usage schema changed')
    result={key:usage[key] for key in ('prompt_tokens','completion_tokens')}
    for key in ('total_tokens','estimated_cost','cost'):
        value=usage.get(key)
        if value is not None:
            if (type(value) not in {int,float} or not math.isfinite(value) or value<0 or
                    (key=='total_tokens' and type(value) is not int)):
                raise ValueError('provider usage contains an invalid numeric field')
            result[key]=value
    for key,allowed in [('prompt_tokens_details',{'cached_tokens','audio_tokens'}),
                        ('completion_tokens_details',{'reasoning_tokens','audio_tokens',
                         'accepted_prediction_tokens','rejected_prediction_tokens'})]:
        details=usage.get(key)
        if details is not None:
            if not isinstance(details,dict):
                raise ValueError('provider token details changed schema')
            selected={k:v for k,v in details.items() if k in allowed and v is not None}
            if any(type(v) is not int or v<0 for v in selected.values()):
                raise ValueError('provider token details are invalid')
            result[key]=selected
    return result


def clean_worker_metrics(value):
    """Validate before publishing worker data or scheduling any retry."""
    allowed={'http_attempts_started','http_status','retry_after_s','usage','finish_reason',
             'system_fingerprint','worker_wall_s','provider','failure_code'}
    if not isinstance(value,dict) or set(value)-allowed:
        raise ValueError('HTTP worker metrics schema changed')
    if type(value.get('http_attempts_started')) is not int or value['http_attempts_started'] not in {0,1}:
        raise ValueError('HTTP worker attempt count is invalid')
    result={'http_attempts_started':value['http_attempts_started'],'usage':clean_usage(value.get('usage'))}
    if 'failure_code' in value:
        if value['failure_code'] not in {'model_refusal','provider_credit_exhausted','provider_authorization_failed','provider_rate_limited','provider_context_exceeded'}:
            raise ValueError('invalid classified provider failure')
        result['failure_code']=value['failure_code']
    for key in ('retry_after_s','worker_wall_s'):
        number=value.get(key)
        if number is not None and (type(number) not in {int,float} or not math.isfinite(number) or number<0):
            raise ValueError('HTTP worker duration is invalid')
        result[key]=number
    status=value.get('http_status')
    if status is not None and (type(status) is not int or not 100<=status<=599):
        raise ValueError('HTTP worker status is invalid')
    result['http_status']=status
    finish=value.get('finish_reason')
    if finish not in {None,'stop','length','tool_calls','content_filter','function_call'}:
        raise ValueError('HTTP finish reason is invalid')
    result['finish_reason']=finish
    fingerprint=value.get('system_fingerprint')
    if fingerprint is not None and (not isinstance(fingerprint,str) or len(fingerprint)>256):
        raise ValueError('HTTP provider fingerprint is invalid')
    result['system_fingerprint']=fingerprint
    provider=value.get('provider')
    if provider is not None:
        if not isinstance(provider,str) or not 1<=len(provider)<=256:
            raise ValueError('HTTP serving provider is invalid')
        result['provider']=provider
    return result
