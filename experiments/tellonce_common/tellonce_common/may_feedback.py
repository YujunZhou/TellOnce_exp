"""May V13d continuous four-step simulator with reviewed task adapters.

The original Chinese prompts and generation settings are embedded from the
pinned May source; no Git or historical user records are read at runtime.
Step four only supplements private labels. Only step three's natural utterance
can enter the public interaction and subsequent memory-learning flow.
"""
import json
import re

from .compact_profiles import render_may_prefix
from .feedback_agent import FeedbackDecision
from .provenance import content_hash
from .text_request_journal import TextAttemptError
from .may_cooccurrence import MAY_COOCCURRENCE

VERSION = 'may-v13d-production-adapter-evidence-scope-v11'   # v11 (2026-10-06): preference_message separates the user's own preference from the task-error part of a mixed correction; only it is learned
# v10 (2026-10-03): listing attributes (minimum nights, occupancy, availability, unrequested house rules) never create requirements
# v9 (2026-10-02): evidence-support / attribution / option-choice judgments are never corrections, however phrased
PROTOCOL = VERSION
MAX_TOKENS = (500, 400, 600, 500)
TEMPERATURES = (.3, .2, .3, .2)

ORIGINAL_DECISION_GATE = '''- matched entry.trigger_type == "correction" 且 task_progress_judgment in ["off_track","stalled"] → should_correct=true
- matched entry.trigger_type == "silent_progression" 或 task_progress_judgment == "on_track" → should_correct=false
- no match 或 completed → should_correct=false'''

ADAPTED_DECISION_GATE = '''- 先核对当前公开事实，以及画像中适用偏好和当前授权/例外；情景匹配帮助理解习惯，不是是否开口的硬门槛。
- 真实适用偏好未满足，或公开事实明确证实任务错误 → should_correct=true；completed/on_track 或 no match 本身不能排除这些问题。
- matched entry 为 silent_progression 时参考其接受条件；只有当前行为确实满足该条件、且没有其他已证实问题时才沉默。
- 没有已证实且可操作的问题 → should_correct=false；不强迫纠正，也不把未知当错误。'''

STEP3_EXCEPTION = '''

# 本次已确认的必要适配（其余原五月判断流程保留）
task_progress_judgment 说任务完成或正常推进，并不自动表示个人偏好已满足。
若当前公开行为确实违反 Layer 2 中适用且未被当前例外覆盖的个人偏好，
即使 completed/on_track 或没有明显匹配的 Layer 3 条目，也应 should_correct=true。
Layer 3 的 silent_progression 仍用于判断什么时候应当接受，不因表面相似强行纠正。
只有可观察的实际违背才适用上述例外，不把人物设定、当前题目要求或缺失片段
编成自己的长期偏好，也不能仅为增加 rule_ids 而给出纠正。
另一个已确认例外：用户也关心任务是否真正解决。当前公开材料能明确证实的
任务错误，即使没有对应个人偏好/Layer 3 匹配，也可自然指出并要求修正。
任务错误只指能用当前请求或公开材料中的明文直接核对的事实没有满足：请求明文规定的
条件（评分、价格、菜系、房型、房屋规则、指定恢复项等）、指定的文件是否产出、明文的格式
要求。公开材料只用来核对请求明文条件是否满足，它本身不产生要求：请求没有明文规定的属性
（房源最大入住人数与同行人数的关系、最低入住晚数与行程天数的关系、是否可订、请求未提的
房屋规则）不是任务错误，即使资料里写明了也一样；不要据此要求更换，不要说成"不适合这次
安排"，也不要把它学成长期要求。首位旅客或其他旅客的条件（包括他们已确认
行程里的房屋规则、房型要求）不约束当前旅客，除非当前请求自己提出；不要以"已确认行程
要求…而你没有要求取消"为由要求更换沿用的项目。"某陈述是否有证据支持""哪些选项正确"
"某来源算不算独立佐证"这类需要综合判断的结论不是可直接核对的事实：不要以你自己的判断
为由要求更换答案，更不要把这种判断学成长期规则；你只纠正习惯和可直接核对的事实。
这条边界不因措辞而变："请核对原文/来源有没有这层意思""原文没说 X，不要把 X 列为支持项""这一项不该选/漏选了"
"把 X 归于来源缺少依据"都仍是你对证据是否充分的判断，哪怕你引用了原文片段也一样，不构成纠正；
只有请求本身明文写出的条件、是否产出了指定文件、明文的格式要求，才是可直接核对的事实。
纯任务错误不伪造个人规则编号；没有真实 Layer 2 违背时 rule_ids_triggered=[]。
本人 Layer 3 条件中明确记录的偏好同样有效；若它没有 Layer 2 编号，可自然
纠正并令 rule_ids_triggered=[]，不要为了私有标签缺失而沉默或编造规则编号。
未知信息或没有选中的片段不是已证实任务错误。以上明确例外优先于旧版
no match/completed/on_track 就绝对沉默的门控，其他原五月流程保持。
纯任务纠正的机械门槛：若 should_correct=true 且 rule_ids_triggered 为空，JSON 还必须含
request_basis——从当前请求原文逐字摘录的一段（至少 6 个字符），写明该条件是请求明文规定的；
系统会逐字核对，核不上就当作没有纠正。引用不到请求原文的"问题"（查询结果里的属性、你对
合适与否的判断）不要纠正。偏好纠正（rule_ids_triggered 非空）不需要 request_basis，可省略或留空。
偏好纠正（rule_ids_triggered 非空）的 JSON 还必须含 preference_message：把 user_would_say 里属于你本人偏好
（所列 Layer 2 条目）的那些句子逐字照抄、每句一行；任务错误的句子（请求明文条件没满足）不要抄进来。系统会
逐字核对；记忆只学 preference_message，整条 user_would_say 仍会发给 Agent。rule_ids_triggered 为空时省略或留空。'''

STEP1_OBSERVATION = '''

# 当前公开观察的必要补充
仍按用户的心理与目标判断，但不要把“其他应该照旧”当成已经核查的一致。
若提供了前后内容或差异记录，逐项看当前请求是否授权该变动；已请求的变化
不是错误，无关变化不能因为主要改动正确而跳过。当前格式是否满足要看实际
公开交付文本；不能仅因内容大致正确就声称满足全部格式。任务事实与个人偏好
分别判断，不把题目要求变成个人习惯；适用范围按每条画像的有效条件和例外。
在既有 user_thinking/progress_reason 字段里，具体“改动/遗漏/不符合”的判断
要引用当前请求及实际 before/after 或交付的短片段，先看证据再作心理判断。
若给出差异列表，核对其中每个变化是否在本次请求范围；按指定基准人物、日期
和字段比较，不能用被提及的另一个人的其他字段替代指定继承基准。不要把
“其他应该一样”说成已经比对一致；未检查的部分保持未知，不声称已全部满足。
'''

BOUNDARY = '''
# 当前任务的轻量适配边界
下面只给本任务用户可见材料，不含标准答案或内部思考过程。题目/文件中的人物
不是正在扮演的用户；人物的要求属于当前任务，不能自动成为你自己的长期偏好。
公开工具和源文件若只给片段，缺片段不是没做、没读或文件不存在的证明。
当前交付文件也是回答；完成的文件内容不需要在最后一句回复中重复一遍。
第一段说明/先讨论后操作的规则要看本任务最早的公开回复和操作顺序，而非只看
最后的总结。若公开 commentary 覆盖不完整，不能声称此前从未说明；不读取或
索要 thinking、analysis、CoT。A4 的内部理解不能凭隐藏过程不可见就判违规。
公开 commentary 的 coverage.complete 只描述该有界投影已记录的大小或工具次序
遗漏，不保证所有公开交流均被捕获。before_first_tool 只相对已观察的工具事件；
若工具开始事件缺失，不能据它证明说明早于实际执行。即使 complete=true 且 messages=[]，也不能仅据此
断言从未有单独开场：没有阶段标记且后面未跟工具的文本可能未进入该投影，
其是否公开可见仍未知；参照 visible_commentary_scope，不索要隐藏思考。
已捕获的公开说明仍是证据，仍须完整阅读实际 delivery；若公开事实足以确定
交付就是首条公开回复，其中的目标/方案/风险说明可以满足首段要求。涉及必须
先说明再操作的条件时，只按可确认的公开顺序判断，不把事后说明当事前说明。
证据不足的先后顺序保持未知，不等于已满足；是否需开场仍按画像适用条件判断。
若有公开旅行继承契约，按该契约和用户当次修改范围判断；不要把通用模板的
Flight 等栏目自动当成本轮必填。当前授权和规则中明确例外优先，保持已满足部分。
画像里的历史情景与心理描述是背景，不是本题已发生行为；说话时用自然的新表达。
单列的 transfer_examples 只是规则迁移参考，不是历史真实纠正或当前事实，
原 Layer 3 索引保留，新增迁移正反情景仅按附加候选目录分配的后续索引匹配。
起草建议、写待批准计划或描述未来承诺，不等于实际执行或对外承诺；本次任务
授权的起草无需重复审批。完整阅读当前交付，不能把已有但不够充分的讨论说成完全没有。
'''


_ORIGINAL_PROMPTS = {'first_three': ['# Step 1/4 - 内想 (只描述脑子里, 不说话)\n'
                 '\n'
                 '基于 profile + 对话 + agent 输出, 输出你的内部状态:\n'
                 '1. **user_thinking**: 心里在想什么 (2-4 句, 第一人称, 连续思考流)\n'
                 '2. **task_progress_judgment**: "on_track" / "off_track" / "stalled" / "completed"\n'
                 '3. **progress_reason**: 一句 why\n'
                 '\n'
                 '严格 JSON:\n'
                 '```json\n'
                 '{"user_thinking": "...", "task_progress_judgment": "...", "progress_reason": "..."}\n'
                 '```',
                 '# Step 2/4 - 情景匹配\n'
                 '\n'
                 '基于前步内想, 匹配当前情景到 Layer 3 里最近的 conditional entry. 如果都不明显匹配, matched_entry_index=null.\n'
                 '\n'
                 '严格 JSON:\n'
                 '```json\n'
                 '{"situation": "1-2句描述当前 agent 做了啥", "matched_entry_index": N or null, "match_confidence": '
                 '0-1, "match_reason": "..."}\n'
                 '```',
                 '# Step 3/4 - 决定 + 说话 + 列全 rule_ids\n'
                 '\n'
                 '基于前两步, 决定 user 是否出手 + 生成说什么.\n'
                 '\n'
                 '规则:\n'
                 '- matched entry.trigger_type == "correction" 且 task_progress_judgment in '
                 '["off_track","stalled"] → should_correct=true\n'
                 '- matched entry.trigger_type == "silent_progression" 或 task_progress_judgment == '
                 '"on_track" → should_correct=false\n'
                 '- no match 或 completed → should_correct=false\n'
                 '\n'
                 '**重要 1**: 若 should_correct=true, user_would_say 必须 (1) 指出具体错误 (2) 给出**正确方向**.\n'
                 '\n'
                 '**重要 2**:\n'
                 '- 扫全 Layer 2 的 rule 列表, 不限于 matched entry 的 related_rule_ids\n'
                 '- 列出 agent 违反的**所有** rule_id (通常 2-4 条)\n'
                 '\n'
                 '严格 JSON:\n'
                 '```json\n'
                 '{"should_correct": true/false, "user_would_say": "..." or null, "rule_ids_triggered": '
                 '["rule_id1", ...], "reasoning": "..."}\n'
                 '```'],
 'fourth_parts': ['# Step 4/4 - miss-check\n\nStep 3 给的 rule_ids_triggered = ',
                  None,
                  '.\n'
                  '\n'
                  '再扫 Layer 2 其他 rule, 是否也被违反? 加到 additional_rules.\n'
                  '\n'
                  '严格 JSON:\n'
                  '```json\n'
                  '{"additional_rules": [...], "reasoning": "...", "final_rule_ids": [...]}\n'
                  '```'],
 'header': '你在扮演 AI 研究员 user. 下面是你的 profile + 对话历史 + agent 最新输出. 接下来 4 步对话里你会按顺序完成 4 个任务.\n'
           '\n'
           '# 用户特征 (simulator fidelity 关键约束 — N5)\n'
           '你不一定懂底层技术细节 (e.g. 不一定知道 LLM internal / API 参数 / 库 implementation / agent harness 工作原理), 但你对自己 '
           'task 的 vision 是清楚的 (大致目标 / 期望产出形态 / 你想 deliver 啥). 用**日常用户语言**, 不发明缩写, 不用陌生 jargon (例如不说 '
           '"fire"/"honored"/"module N"/"hook"/"oracle"/"force-comply" 等系统内部词). 写出来要像真人用户给 AI agent 提问 / '
           '纠正的口语, 不像 AI 写的技术 spec.\n'
           '\n',
 'source': {'commit': '1df0774',
            'path': 'experiment/simulated_user/v13d_variants.py',
            'sha256': 'bbcbea344f4aa0b15ed02e99d7fbad48e6862dccea1f60fd07d597478b241cbf'}}

def original_prompts():
    return _ORIGINAL_PROMPTS

def fourth_prompt(rule_ids):
    return ''.join(str(rule_ids) if part is None else part for part in original_prompts()['fourth_parts'])


def transfer_candidates(profile):
    """Append authorized examples after historical entries without renumbering."""
    start = len(profile['layer_3']['conditional_entries'])
    candidates = []
    for example in profile.get('transfer_examples', []):
        for key, trigger in (('correction_case', 'correction'),
                             ('no_correction_case', 'silent_progression')):
            candidates.append({'matched_entry_index': start + len(candidates),
                'source': 'authorized_transfer_reference', 'example_id': example['example_id'],
                'trigger_type': trigger, 'situation': example['task'],
                'agent_behavior': example[key]['agent_behavior'],
                'related_rule_ids': example['source_rule_ids']})
    return candidates


def step_prompt(stage, profile, state):
    if stage == 4:
        return fourth_prompt(state['step3']['rule_ids_triggered'])
    prompt = original_prompts()['first_three'][stage - 1]
    if stage == 1:
        return prompt + STEP1_OBSERVATION
    if stage == 2:
        return prompt + '''\n\n本次已批准跨任务适配：除原 Layer 3 条目，也可匹配系统中附加目录里的
迁移正反情景。使用目录明确给出的 matched_entry_index，原条目序号不变。
按行为及适用条件选择最接近的一条（包括无需纠正的情景），不要只按题材词匹配。
这些都是参考场景，不是当前行为证据；都不明显匹配时仍可为 null。'''
    if ORIGINAL_DECISION_GATE not in prompt:
        raise ValueError('historical decision gate changed; review the adaptation')
    # Replace, rather than append an instruction contradicting the old gate.
    return prompt.replace(ORIGINAL_DECISION_GATE, ADAPTED_DECISION_GATE, 1) + STEP3_EXCEPTION


def build_prefix(profile_prefix, public, delivery):
    if not isinstance(profile_prefix, str) or not profile_prefix.strip() or not isinstance(public, dict) or not isinstance(delivery, str):
        raise ValueError('complete May profile and public current turn required')
    # Check explicit private containers, never infer a public task's answer key.
    pending = [public]
    forbidden = {'training_reference', 'private_evaluation', 'hidden_score', 'reference_answer', 'gold_answer', 'user_profile'}
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError('private/reference/foreign-persona field in public turn')
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    from .operation_scope import SCOPE_INSTRUCTIONS
    context = json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return (original_prompts()['header'] + '\n# 完整五月画像（保留三层、纠正与沉默例）\n' + profile_prefix +
            BOUNDARY + '\n' + SCOPE_INSTRUCTIONS + '\n# 当前任务的真实公开上下文\n' + context +
            '\n# Agent 最新完整交付\n' + delivery +
            '\n---\n以上是你的 context。下面按五月原来的四步答，每步只输出该步骤的 JSON 字段组。')


def _object(text):
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError('bounded stage JSON text required')
    text = text.strip()
    match = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', text, flags=re.DOTALL)
    if match:
        text = match[1]
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate stage field')
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('stage must return a JSON object')
    return value


def _text(value, allow_empty=False):
    if not isinstance(value, str) or len(value) > 5000 or (not allow_empty and not value.strip()):
        raise ValueError('bounded stage text required')


def _rules(value, known):
    if not isinstance(value, list) or any(not isinstance(r, str) or r not in known for r in value):
        raise ValueError('only actual May Layer 2 rule IDs are allowed')
    if len(value) != len(set(value)):
        raise ValueError('duplicate rule ID')


REQUEST_BASIS_MIN_CHARS = 6


def _normalize_request_text(text):
    return ' '.join(str(text).split()).casefold()


def preference_message_check(value):
    """A preference correction names its own preference sentences; only those are learned (user decision 2026-10-04:
    a task's stated condition is not the user's rule). Verbatim lines of user_would_say; '' for a task-only correction."""
    preference = value.get('preference_message') or ''
    if not isinstance(preference, str):
        raise ValueError('preference_message must be text')
    correction = value['should_correct'] and bool(value['rule_ids_triggered'])
    if correction and not preference.strip():
        raise ValueError('a preference correction must quote its preference sentences in preference_message')
    if not correction:
        return ''   # a task-only correction (or silence) carries no learning sentences, whatever was written
    lines = [line.strip() for line in preference.splitlines() if line.strip()]
    if any(line not in (value['user_would_say'] or '') for line in lines):
        raise ValueError('preference_message must consist of verbatim lines of user_would_say')
    return preference


def request_basis_gate(value, request_text):
    """A pure task correction (no Layer 2 rule id) must quote the request it rests on.

    2026-10-03 round 19: the corrector replaced a gold-matching stay for a listing's 7-night minimum that the request
    never mentioned; the learner generalised it and the checker then failed 86 % of accommodation deliveries. Lookup
    results and listings can show that a stated condition is unmet; they cannot create one. The decision is recorded
    as silence with the withheld message kept for audit — deterministic, no further model call."""
    if not value.get('should_correct') or value.get('rule_ids_triggered'):
        return value
    basis = value.get('request_basis')
    if (isinstance(basis, str) and len(basis.strip()) >= REQUEST_BASIS_MIN_CHARS
            and _normalize_request_text(basis) in _normalize_request_text(request_text)):
        return value
    return {**value, 'should_correct': False, 'user_would_say': None,
            'reasoning': value['reasoning'] + ' [request-basis gate: a task correction must quote the request; '
                         + ('no excerpt given' if not basis else 'excerpt not found in the request') + ']',
            'request_basis_gate': 'downgraded_no_request_citation', 'withheld_message': value['user_would_say']}


def parse_stage(text, stage, profile, request_text=None):
    value = _object(text)
    known = {row['rule_id'] for row in profile['layer_2_preferences']}
    if stage == 1:
        if set(value) != {'user_thinking', 'task_progress_judgment', 'progress_reason'}:
            raise ValueError('May step 1 schema mismatch')
        _text(value['user_thinking']); _text(value['progress_reason'])
        if (not isinstance(value['task_progress_judgment'], str) or
                value['task_progress_judgment'] not in {'on_track', 'off_track', 'stalled', 'completed'}):
            raise ValueError('invalid May progress label')
    elif stage == 2:
        if set(value) != {'situation', 'matched_entry_index', 'match_confidence', 'match_reason'}:
            raise ValueError('May step 2 schema mismatch')
        _text(value['situation']); _text(value['match_reason'])
        index, confidence = value['matched_entry_index'], value['match_confidence']
        candidate_count = len(profile['layer_3']['conditional_entries']) + len(transfer_candidates(profile))
        if index is not None and (type(index) is not int or not 0 <= index < candidate_count):
            raise ValueError('matched entry must belong to the actual May profile')
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError('bounded numerical match confidence required')
    elif stage == 3:
        if set(value) - {'request_basis', 'preference_message'} != {'should_correct', 'user_would_say', 'rule_ids_triggered', 'reasoning'}:
            raise ValueError('May step 3 schema mismatch')
        if type(value['should_correct']) is not bool:
            raise ValueError('actual Boolean correction decision required')
        _text(value['reasoning']); _rules(value['rule_ids_triggered'], known)
        if value['should_correct']:
            _text(value['user_would_say'])
        elif value['user_would_say'] not in (None, ''):
            raise ValueError('silent May decision cannot contain a user correction')
        value = {**value, 'preference_message': preference_message_check(value)}
        if 'request_basis' in value and value['request_basis'] not in (None, '') and not isinstance(value['request_basis'], str):
            raise ValueError('request_basis must be a literal request excerpt')
        if request_text is not None:
            value = request_basis_gate(value, request_text)
    elif stage == 4:
        if set(value) != {'additional_rules', 'reasoning', 'final_rule_ids'}:
            raise ValueError('May step 4 schema mismatch')
        _text(value['reasoning']); _rules(value['additional_rules'], known); _rules(value['final_rule_ids'], known)
    else:
        raise ValueError('unknown stage')
    return value


def cooccurrence_expand(seed_rules, rule_count, cooccur, layer2_ids):
    """Original V13d threshold/order/cap behavior, private labels only."""
    seed = list(seed_rules) if seed_rules else []
    if not seed:
        return []
    current, score = set(seed), {}
    for rule in seed:
        base = max(int(rule_count.get(rule, 1)), 1)
        for other, count in cooccur.get(rule, {}).items():
            if other in current or other not in layer2_ids:
                continue
            ratio = count / base
            if ratio >= .30:
                score[other] = score.get(other, 0.) + ratio
    additions = sorted(score.items(), key=lambda row: -row[1])
    return seed + [rule for rule, _ in additions[:max(0, 5 - len(current))]]


def run_four_steps(profile, prefix, chat, *, cooccurrence, request_text=None):
    candidates = transfer_candidates(profile)
    if candidates:
        prefix += ('\n# 新批准迁移参考的附加匹配候选（非历史实录或当前证据）\n' +
                   json.dumps(candidates, ensure_ascii=False, sort_keys=True))
    messages = [{'role': 'system', 'content': prefix}]
    state = {}
    for index in range(1, 5):
        prompt = step_prompt(index, profile, state)
        messages.append({'role': 'user', 'content': prompt})
        text = chat(messages, stage=index, max_tokens=MAX_TOKENS[index - 1], temperature=TEMPERATURES[index - 1])
        value = parse_stage(text, index, profile, request_text)
        state['step' + str(index)] = value
        messages.append({'role': 'assistant', 'content': text})
    third, fourth = state['step3'], state['step4']
    # The validated final list is authoritative, including an explicit retraction.
    rules = list(fourth['final_rule_ids'])
    expanded = (cooccurrence_expand(rules, cooccurrence['rule_count'], cooccurrence['cooccur'],
                                   {row['rule_id'] for row in profile['layer_2_preferences']})
                if third['should_correct'] else list(rules))
    return {'should_correct': third['should_correct'], 'correction_text': third['user_would_say'],
            'preference_message': third.get('preference_message', '') if third['should_correct'] and third['rule_ids_triggered'] else '',
            'request_basis_gate': third.get('request_basis_gate'), 'withheld_message': third.get('withheld_message'),
            'rule_ids_triggered': expanded, 'step3_rule_ids': third['rule_ids_triggered'],
            'step4_final_rule_ids': rules, 'step4_added': fourth['additional_rules'],
            'cooccurrence_added': [rule for rule in expanded if rule not in rules], 'full_state': state}


def request_text_of(view):
    """The user's own request text: the current instruction plus any prior requests of this user. Other
    travellers' or other users' turns (visible_history) do not bind the current request."""
    parts = []
    for key in ('instruction', 'latest_user_request'):
        if isinstance(view.get(key), str):
            parts.append(view[key])
    prior = view.get('prior_user_requests')
    if isinstance(prior, dict):
        prior = prior.get('messages')
    if isinstance(prior, list):
        parts += [m.get('text', '') for m in prior if isinstance(m, dict) and isinstance(m.get('text'), str)]
    return '\n'.join(parts)


def run_feedback(view, transport, context):
    """Run original May stages through the admitted durable conversation API.

    Recovered stages replay without another HTTP request. Only an already failed
    stage uses the interaction owner's second technical attempt. False means no
    correction was emitted; independent scorers determine actual task/preference
    success. The private summary is compatibility metadata, never future evidence.
    """
    envelope = view.get('user_profile', {})
    profile = envelope.get('simulator_profile')
    profile_prefix = render_may_prefix(profile)
    public = {key: value for key, value in view.items() if key not in {'user_profile', 'delivery'}}
    prefix = build_prefix(profile_prefix, public, view['delivery'])
    def chat(messages, *, stage, max_tokens, temperature):
        stage_context = {**context, 'call_id': content_hash([context['call_id'], PROTOCOL, stage]), 'attempt': 1}
        generation = {'max_tokens': max_tokens, 'temperature': temperature}
        validator = lambda text: parse_stage(text, stage, profile, request_text_of(view))
        try:
            reply = transport.call_messages(messages, context=stage_context,
                                            generation=generation, validate=validator)
        except TextAttemptError as exc:
            if context['attempt'] != 2 or not exc.replayed or not exc.retryable:
                raise
            reply = transport.call_messages(messages, context={**stage_context, 'attempt': 2},
                                            generation=generation, validate=validator)
        return reply.text
    result = run_four_steps(profile, prefix, chat, cooccurrence=MAY_COOCCURRENCE[envelope['user_id']],
                            request_text=request_text_of(view))
    record = getattr(transport, 'record', None)
    if callable(record):
        record({'type': 'may_feedback_private_labels', 'protocol': PROTOCOL, 'context': dict(context),
                **{key: result[key] for key in ('should_correct', 'rule_ids_triggered', 'request_basis_gate',
                   'step3_rule_ids', 'step4_final_rule_ids', 'step4_added', 'cooccurrence_added')}})
    # May uses complete same-task public history on its next turn, not a model
    # summary. A fixed compatibility value keeps the existing checkpoint schema.
    learnable = not result['should_correct'] or bool(result['step3_rule_ids'])
    return FeedbackDecision('correction' if result['should_correct'] else 'satisfied',
                            result['correction_text'] or '', 'May decision recorded; use original public task history.',
                            learnable=learnable, learning_message=result['preference_message'] if learnable else '')
