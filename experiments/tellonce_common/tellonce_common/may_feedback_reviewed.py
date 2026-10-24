"""Opt-in May final-decision review; legacy v4 receipts remain replayable.

The first three stages are unchanged. The fourth can retract or replace their
draft using public evidence. Exact citations prove provenance, not semantic
correctness; independent annotations are still required before live admission.
"""
import json
import re

from . import may_feedback as legacy
from .compact_profiles import render_may_prefix
from .feedback_agent import FeedbackDecision
from .provenance import content_hash
from .text_request_journal import TextAttemptError

PROTOCOL = 'may-v13d-final-decision-review-v11'   # v11 (2026-10-06): preference_message separates the user's own preference from the task-error part of a mixed correction; only it is learned
# v10 (2026-10-03): a pure task correction must cite the request itself; listing attributes never create requirements
# v9 (2026-10-02): evidence-support / attribution / option-choice judgments are never corrections, however phrased
REQUEST_ROOTS = frozenset({'instruction', 'latest_user_request', 'prior_user_requests'})
PUBLIC_ROOTS = frozenset({
    'instruction', 'latest_user_request', 'delivery', 'visible_commentary',
    'visible_history', 'current_delivered_files', 'public_context',
    'public_evidence', 'public_lookup', 'unreviewed_outputs', 'prior_user_requests',
})

REVIEW_PROMPT = '''# Step 4/4 - 公开证据复核与最终决定
前三步是可错的草稿，不是证据。这一步可以保留、替换或撤回纠正，也可以在
此前沉默时指出现在证实的问题。最终交付和后续学习只使用这一步的决定。

重新核对完整画像的适用条件/当前授权/例外，以及实际公开内容。历史情景只是
参考；不要将特定“改代码前等待批准”扩大为所有分析答题都必须再等批准。
先读 visible_commentary 中实际最早公开回复及 before_first_tool、coverage，
再看 delivery。已有但不充分的开场不能说成完全没有。只有缺片段时保留未知，
也不能把完整空窗口自动当违规。题目人物的习惯不是当前用户的长期偏好。
任务正确性与个人偏好分开判断：忠实引用源数字不保证其运算正确；不得凭
前三步的“准确”跳过实际交付。只纠正公开材料支持的问题，不使用隐藏答案。
任务问题只指能用当前请求或公开材料中的明文直接核对的事实没有满足：请求明文规定的
条件（评分、价格、菜系、房型、房屋规则、指定恢复项等）、指定文件是否产出、明文格式要求。
请求没有明文规定的属性（例如房源最大入住人数与同行人数的关系）不是任务问题，不要据此
要求更换。首位旅客或其他旅客的条件（包括他们已确认行程里的房屋规则、房型要求）不约束
当前旅客，除非当前请求自己提出。"某陈述是否有证据支持""哪些选项正确""某来源算不算独立
佐证"这类综合判断不是可直接核对的事实，不作为任务问题，不要据此要求更换答案。措辞成"请核对原文有没有这层意思"
"原文没说 X 就不要把 X 列为支持项""这一项不该选/漏选""归于来源缺少依据"也一样是综合判断，即使引用了原文片段，
也不能作为 correction；前三步若给出了这类纠正，这一步必须撤回它。

status 为 correction / satisfied / cannot_judge：
- correction：有具体、可操作且证据支持的问题；message 用自然用户语言
  说明问题和方向，保留已满足部分。rule_ids 仅列真正违反的画像 Layer 2 编号；
  纯任务问题或无编号的条件偏好可以为空，禁止按共现推测更多违规。
- satisfied：公开工作足以支持请求已解决、适用偏好满足；message=''，rule_ids=[]。
- cannot_judge：必要证据不足，且不能指出已证实的可操作问题；message=''，rule_ids=[]。
  撤回无依据的纠正不自动等于满意。不强迫纠正，不把沉默变成成功标签。

evidence 引用系统中“当前任务的真实公开上下文”或“Agent 最新完整交付”：
pointer 是对应 JSON 字段的绝对 JSON Pointer；交付用 /delivery，开场例如
/visible_commentary/messages/0/text，历史例如 /visible_history/0/text。
对象键中的 ~ 编为 ~0，/ 编为 ~1。quote 必须是该公开字符串中的非空原文。
不能引用画像、前三步判断、previous_feedback、模型摘要、内部推理或私有评分
作为实际行为证据。引用只能证明原文存在，理由仍须核对其意义与适用范围。
correction/satisfied 至少引用一处直接相关证据；cannot_judge 可为空。
reasoning 用简短可核对的理由；建议 1–2 处短引文，不重抄整个回答。
纯任务问题（rule_ids 为空的 correction）必须在 evidence 中至少引用一处当前请求原文
（pointer 以 /instruction、/latest_user_request 或 /prior_user_requests 开头），证明该条件是请求
明文规定的；只引用查询结果、房源资料、交付或开场而不引用请求的任务纠正，会被系统判为
cannot_judge。公开材料只用于核对请求条件是否满足：资料里写明的属性（最低入住晚数、可住
人数、是否可订、请求未提的房屋规则）不构成任务问题，不要据此要求更换，也不要把它说成
"不适合这次安排"。

preference_message：message 里属于你本人偏好（rule_ids 所列 Layer 2 条目）的那些句子，逐字照抄、
每句一行；任务问题的句子（请求明文条件没满足）不要抄进来。rule_ids 非空时必须非空；rule_ids 为空
时必须是 ""。记忆只学 preference_message，整条 message 仍会发给 Agent。

严格输出一个 JSON 对象，不增加字段：
{"status":"...","message":"...","rule_ids":[],"preference_message":"...","reasoning":"...",
 "evidence":[{"pointer":"/delivery","quote":"原文短片段"}]}
'''


def _public_quote(view, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/') or len(pointer) > 1000:
        raise ValueError('bounded public JSON pointer required')
    raw = pointer[1:].split('/')
    if any(re.search(r'~(?![01])', part) for part in raw):
        raise ValueError('invalid JSON pointer escape')
    parts = [part.replace('~1', '/').replace('~0', '~') for part in raw]
    if parts[0] not in PUBLIC_ROOTS:
        raise ValueError('review evidence must cite original public observations')
    value = view
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and re.fullmatch(r'0|[1-9][0-9]*', part):
            index = int(part)
            if index >= len(value):
                raise ValueError('public evidence index out of bounds')
            value = value[index]
        else:
            raise ValueError('public evidence pointer does not resolve')
    if not isinstance(value, str):
        raise ValueError('public evidence must resolve to original text')
    return value


def parse_review(text, view):
    value = legacy._object(text)
    if set(value) != {'status', 'message', 'rule_ids', 'preference_message', 'reasoning', 'evidence'}:
        raise ValueError('final feedback review schema mismatch')
    # FeedbackDecision validates status/message coherence without learning any
    # review metadata or converting unknown to satisfaction.
    FeedbackDecision(value['status'], value['message'])
    preference = value['preference_message']
    if not isinstance(preference, str):
        raise ValueError('preference_message must be text')
    if value['rule_ids'] and not preference.strip():
        raise ValueError('a preference correction must quote its preference sentences in preference_message')
    if not value['rule_ids']:
        value = {**value, 'preference_message': ''}   # a task-only correction carries no learning sentences
    lines = [line.strip() for line in value['preference_message'].splitlines() if line.strip()]
    if any(line not in value['message'] for line in lines):
        raise ValueError('preference_message must consist of verbatim lines of message')
    legacy._text(value['message'], allow_empty=True)
    legacy._text(value['reasoning'])
    known = {row['rule_id'] for row in view['user_profile']['simulator_profile']['layer_2_preferences']}
    legacy._rules(value['rule_ids'], known)
    if value['status'] != 'correction' and value['rule_ids']:
        raise ValueError('non-correction cannot assert unspoken violations')
    evidence = value['evidence']
    if not isinstance(evidence, list) or len(evidence) > 8:
        raise ValueError('bounded evidence list required')
    if value['status'] != 'cannot_judge' and not evidence:
        raise ValueError('a definite feedback decision requires public evidence')
    seen = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {'pointer', 'quote'}:
            raise ValueError('public pointer and literal quote required')
        quote = item['quote']
        if not isinstance(quote, str) or not quote.strip() or len(quote) > 400:
            raise ValueError('bounded nonempty literal quote required')
        original = _public_quote(view, item['pointer'])
        if quote not in original:
            raise ValueError('review quote absent from cited public text')
        key = (item['pointer'], quote)
        if key in seen:
            raise ValueError('duplicate public evidence citation')
        seen.add(key)
    return request_basis_gate(value)


def _pointer_root(pointer):
    return pointer[1:].split('/')[0].replace('~1', '/').replace('~0', '~')


def request_basis_gate(value):
    """A task correction (no Layer 2 rule) must cite the request itself.

    Lookup results, listings, the delivery or the opening can show that a stated
    condition is unmet, but they cannot create a requirement; a correction that
    cites none of the request roots is recorded as cannot_judge, deterministically
    and without another call, so the loop and the learner never see it."""
    if value['status'] != 'correction' or value['rule_ids']:
        return value
    if any(_pointer_root(item['pointer']) in REQUEST_ROOTS for item in value['evidence']):
        return value
    return {**value, 'status': 'cannot_judge', 'message': '', 'rule_ids': [], 'preference_message': '',
            'reasoning': value['reasoning'] + ' [request-basis gate: a task correction must cite the request; none cited]',
            'request_basis_gate': 'downgraded_no_request_citation', 'withheld_message': value['message']}


def run_feedback(view, transport, context):
    profile = view['user_profile']['simulator_profile']
    public = {k: v for k, v in view.items() if k not in {'user_profile', 'delivery'}}
    prefix = legacy.build_prefix(render_may_prefix(profile), public, view['delivery'])
    candidates = legacy.transfer_candidates(profile)
    if candidates:
        prefix += ('\n# 新批准迁移参考的附加匹配候选（非历史实录或当前证据）\n' +
                   json.dumps(candidates, ensure_ascii=False, sort_keys=True))
    messages, state = [{'role': 'system', 'content': prefix}], {}
    for stage in range(1, 5):
        prompt = REVIEW_PROMPT if stage == 4 else legacy.step_prompt(stage, profile, state)
        messages.append({'role': 'user', 'content': prompt})
        validator = (lambda text: parse_review(text, view)) if stage == 4 else (
            lambda text, stage=stage: legacy.parse_stage(text, stage, profile))
        stage_context = {**context, 'call_id': content_hash([context['call_id'], PROTOCOL, stage]),
                         'attempt': 1}
        generation = {'max_tokens': legacy.MAX_TOKENS[stage - 1],
                      'temperature': legacy.TEMPERATURES[stage - 1]}
        try:
            reply = transport.call_messages(messages, context=stage_context,
                generation=generation, validate=validator)
        except TextAttemptError as exc:
            if context['attempt'] != 2 or not exc.replayed or not exc.retryable:
                raise
            reply = transport.call_messages(messages, context={**stage_context, 'attempt': 2},
                generation=generation, validate=validator)
        state['step' + str(stage)] = validator(reply.text)
        messages.append({'role': 'assistant', 'content': reply.text})
    draft, final = state['step3'], state['step4']
    if draft['should_correct'] and final['status'] != 'correction':
        disposition = 'retracted'
    elif final['status'] == 'correction' and not draft['should_correct']:
        disposition = 'added'
    elif final['status'] == 'correction' and final['message'] != draft['user_would_say']:
        disposition = 'replaced'
    else:
        disposition = 'retained'
    record = getattr(transport, 'record', None)
    if callable(record):
        record({'type': 'may_feedback_final_review', 'protocol': PROTOCOL,
                'context': dict(context), 'input_hash': content_hash(view),
                'draft': draft, 'final': final, 'disposition': disposition})
    learnable = final['status'] != 'correction' or bool(final['rule_ids'])
    return FeedbackDecision(final['status'], final['message'],
                            'Final review recorded; use original public task history.',
                            learnable=learnable, learning_message=final['preference_message'] if learnable else '')
