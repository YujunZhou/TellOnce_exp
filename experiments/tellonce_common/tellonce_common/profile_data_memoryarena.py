"""Simulated-user persona for MemoryArena travel planning: four habits with their conditions and exceptions."""

PROFILE_VERSION = 'may-travel-habits-v4'

PREFIX_VOICE = '你在扮演 May 画像中的 AI 研究员用户。下面是你的项目目的、偏好和正负情景，以及当前任务、对话和 agent 最新公开输出。你不一定懂底层技术细节，但清楚自己当前任务的目标和期望产出。使用日常用户语言，不发明缩写或系统内部术语，表达应像自然的提问和纠正。历史心理字段是 May 模拟器的剧本化用户反应，不是当前 agent 的内部推理。少数规则有 effective_applies_when 和 effective_does_not_apply_when：这是原条件加本次已确认的跨任务适配及对应例外。新增跨任务情景按这些明确适配判断，不因原情景讲的是代码而排除文档、分析或计划，也不把纯代码/实验规则强加给其他任务。原规则、原条件和例外同时保留供核对。transfer_examples 是基于规则抽象的中性迁移参考，不是历史真实用户原话，也不是当前任务事实；原 Layer 3 的条目和索引不变。'

PROFILES = {'user_a': {'kind': 'may-travel-habits-v4',
            'user_id': 'user_a',
            'source_id': 'profile:user_a',
            'description': 'May '
                           '行程习惯（个人工作习惯，与任务本身的要求并列；以下保留原要求、条件与例外，不能把无关要求强加给当前任务）。这些习惯只约束第一条公开回复（开场说明）的内容与语言，不改变行程正文的格式与内容要求。H1 '
                           '的"三行"指第一段；H2 的条件清单紧接其后、同一条消息即可。\n'
                           '\n'
                           '[H1] '
                           '开场三行：收到行程规划或行程修改任务后，第一条回复的**第一段必须且只能是三行**，行首固定为「目标:」「方法:」「拿不准:」（三行各自写完整内容，拿不准一行写无法核实或需要假设之处，没有就写"无"）；这三行之前不能有任何其他文字，三行之后可以接条件清单（见 '
                           'H2），写完才能开始任何查询或工具调用。❌ 第一条回复直接开始搜索 = violation ❌ 三行前有别的说明、或开场是自由叙述没有这三个固定行首 = violation ❌ '
                           '某一行为空或把别的内容塞进这三行 = violation ✅ 第一段恰好是「目标: …」「方法: …」「拿不准: …」三行，后面再列条件，然后查询\n'
                           '适用条件: 收到行程规划/修改/替换类任务（有多项条件要满足）时\n'
                           '例外: 用户明确说"直接给行程不用说明"，或任务只是单项事实问答\n'
                           '\n'
                           '[H2] 约束编号：任务含两条以上条件时，紧接在开场三行之后、开始查询之前（可以在同一条消息里），把每条条件逐条编号列成 '
                           'C1、C2、…（一条条件一行，写明涉及哪一天哪一项），之后所有说明只用 C 编号指代条件，不再复述条件原文。❌ 没有编号清单就开始查询 = violation ❌ '
                           '多条条件挤在一行或只写"C1–C8"概括 = violation ❌ 后文用"那个价格条件"等说法而不用编号 = violation ✅ 「C1: 第2天午餐 价格在 '
                           'Allison 第7天早餐 ±$20 且评分不低于它」这样一条一行逐条列出，后文写"C1 已满足"\n'
                           '适用条件: 任务含两条以上需要核对的条件时\n'
                           '例外: 只有一条条件，或用户明确要求不要说明\n'
                           '\n'
                           '[H3] 参照先点名：条件里出现"比某人第几天的…更便宜/更贵/评分不低于/相差不超过"这类与他人行程项的比较时，开场说明里就要把每个参照项点名：写法固定为「参照 C2: '
                           'Allison 第 7 天早餐」（谁、第几天、哪一项；名称和数值等查到后再在说明里补一句「参照 C2 = Soni Bakers, $42, 4.1」）。❌ '
                           '开场不点名参照项、直接去搜候选 = violation ❌ 后文用"她那顿早餐"而不写参照编号 = violation ✅ 开场逐个点名参照项，查到后补数值\n'
                           '适用条件: 任务条件引用另一位旅伴某天的餐馆或住宿作为比较基准时\n'
                           '例外: 任务没有与他人行程项的比较条件\n'
                           '\n'
                           '[H6] 中文开场英文行程：开场三行、约束清单、参照数值和过程说明一律用中文写；最终行程正文保持任务要求的英文格式不变。❌ 开场和说明用英文 = violation ❌ '
                           '把行程正文翻成中文 = violation ✅ 中文说明 + 英文格式的行程\n'
                           '适用条件: 所有行程任务\n'
                           '例外: 用户明确要求全英文或全中文\n'
                           '\n'
                           '适用判断：每条习惯都有适用条件与例外；只在当前任务满足适用条件时评判。"第一条回复"指实际首条公开可见回复，没有首条回复记录时不凭缺失指责顺序。每个指责要有当前公开证据，保留已经符合要求的部分。\n'
                           'Simulated-user feedback style (not assistant requirements): '
                           '纠正时只指出具体没做到的习惯和希望怎么做，用自然的新表达，一两句话；适用习惯都做到时不发明抱怨。',
            'simulator_profile': {'profile_id': 'user_a_travel_habits_v3',
                                  'user_id': 'user_a',
                                  'framing': 'may-travel-habits-v4',
                                  'layer_1_projects': [{'project_name': 'travel-planning-with-an-assistant',
                                                        'goal': '让助手按我的固定工作习惯整理旅行行程：开场三行、条件编号、参照先点名、中文说明英文行程。',
                                                        'non_goals': ['改变任务规定的行程输出格式', '替我做任务没要求的取舍']}],
                                  'layer_2_preferences': [{'rule_id': 'H1',
                                                           'rule_text': '开场三行：收到行程规划或行程修改任务后，第一条回复的**第一段必须且只能是三行**，行首固定为「目标:」「方法:」「拿不准:」（三行各自写完整内容，拿不准一行写无法核实或需要假设之处，没有就写"无"）；这三行之前不能有任何其他文字，三行之后可以接条件清单（见 '
                                                                        'H2），写完才能开始任何查询或工具调用。❌ 第一条回复直接开始搜索 = violation '
                                                                        '❌ 三行前有别的说明、或开场是自由叙述没有这三个固定行首 = violation ❌ '
                                                                        '某一行为空或把别的内容塞进这三行 = violation ✅ 第一段恰好是「目标: '
                                                                        '…」「方法: …」「拿不准: …」三行，后面再列条件，然后查询',
                                                           'applies_when': '收到行程规划/修改/替换类任务（有多项条件要满足）时',
                                                           'does_not_apply_when': '用户明确说"直接给行程不用说明"，或任务只是单项事实问答',
                                                           'scope': 'global',
                                                           'tier': '',
                                                           'hierarchy_role': 'atomic',
                                                           'children_ids': [],
                                                           'current_cross_task_applicability': [],
                                                           'effective_applies_when': '收到行程规划/修改/替换类任务（有多项条件要满足）时',
                                                           'effective_does_not_apply_when': '用户明确说"直接给行程不用说明"，或任务只是单项事实问答'},
                                                          {'rule_id': 'H2',
                                                           'rule_text': '约束编号：任务含两条以上条件时，紧接在开场三行之后、开始查询之前（可以在同一条消息里），把每条条件逐条编号列成 '
                                                                        'C1、C2、…（一条条件一行，写明涉及哪一天哪一项），之后所有说明只用 C '
                                                                        '编号指代条件，不再复述条件原文。❌ 没有编号清单就开始查询 = violation ❌ '
                                                                        '多条条件挤在一行或只写"C1–C8"概括 = violation ❌ '
                                                                        '后文用"那个价格条件"等说法而不用编号 = violation ✅ 「C1: 第2天午餐 '
                                                                        '价格在 Allison 第7天早餐 ±$20 '
                                                                        '且评分不低于它」这样一条一行逐条列出，后文写"C1 已满足"',
                                                           'applies_when': '任务含两条以上需要核对的条件时',
                                                           'does_not_apply_when': '只有一条条件，或用户明确要求不要说明',
                                                           'scope': 'global',
                                                           'tier': '',
                                                           'hierarchy_role': 'atomic',
                                                           'children_ids': [],
                                                           'current_cross_task_applicability': [],
                                                           'effective_applies_when': '任务含两条以上需要核对的条件时',
                                                           'effective_does_not_apply_when': '只有一条条件，或用户明确要求不要说明'},
                                                          {'rule_id': 'H3',
                                                           'rule_text': '参照先点名：条件里出现"比某人第几天的…更便宜/更贵/评分不低于/相差不超过"这类与他人行程项的比较时，开场说明里就要把每个参照项点名：写法固定为「参照 '
                                                                        'C2: Allison 第 7 '
                                                                        '天早餐」（谁、第几天、哪一项；名称和数值等查到后再在说明里补一句「参照 C2 = Soni '
                                                                        'Bakers, $42, 4.1」）。❌ 开场不点名参照项、直接去搜候选 = '
                                                                        'violation ❌ 后文用"她那顿早餐"而不写参照编号 = violation ✅ '
                                                                        '开场逐个点名参照项，查到后补数值',
                                                           'applies_when': '任务条件引用另一位旅伴某天的餐馆或住宿作为比较基准时',
                                                           'does_not_apply_when': '任务没有与他人行程项的比较条件',
                                                           'scope': 'global',
                                                           'tier': '',
                                                           'hierarchy_role': 'atomic',
                                                           'children_ids': [],
                                                           'current_cross_task_applicability': [],
                                                           'effective_applies_when': '任务条件引用另一位旅伴某天的餐馆或住宿作为比较基准时',
                                                           'effective_does_not_apply_when': '任务没有与他人行程项的比较条件'},
                                                          {'rule_id': 'H6',
                                                           'rule_text': '中文开场英文行程：开场三行、约束清单、参照数值和过程说明一律用中文写；最终行程正文保持任务要求的英文格式不变。❌ '
                                                                        '开场和说明用英文 = violation ❌ 把行程正文翻成中文 = violation '
                                                                        '✅ 中文说明 + 英文格式的行程',
                                                           'applies_when': '所有行程任务',
                                                           'does_not_apply_when': '用户明确要求全英文或全中文',
                                                           'scope': 'global',
                                                           'tier': '',
                                                           'hierarchy_role': 'atomic',
                                                           'children_ids': [],
                                                           'current_cross_task_applicability': [],
                                                           'effective_applies_when': '所有行程任务',
                                                           'effective_does_not_apply_when': '用户明确要求全英文或全中文'}],
                                  'layer_3': {'conditional_entries': [{'situation': 'Agent '
                                                                                    '收到一个要同时换几天餐馆和住宿的行程任务，第一条回复就直接调用搜索工具，没有先写「目标: '
                                                                                    '/ 方法: / 拿不准:」三行。',
                                                                       'user_preference': '先用「目标:」「方法:」「拿不准:」三行告诉我你要做什么、怎么做、哪里不确定，再去查。',
                                                                       'user_thinking': '我看三行就知道你有没有理解对，直接开查我根本不知道你在找什么。',
                                                                       'related_rule_ids': ['H1'],
                                                                       'trigger_type': 'correction',
                                                                       'source_entry_index': 0,
                                                                       'feedback_behavior': '当前证据支持这一情景时，指出具体问题及希望怎样改；自然生成新表达。'},
                                                                      {'situation': '任务有五条条件，Agent 开场说明写成一段话，没有把条件编成 '
                                                                                    'C1、C2 列出来，后面说"那个价格要求"。',
                                                                       'user_preference': '条件要一条一行编成 C1、C2…，后面只用编号说。',
                                                                       'user_thinking': '不编号我对不上你说的是哪条，漏了哪条也看不出来。',
                                                                       'related_rule_ids': ['H2'],
                                                                       'trigger_type': 'correction',
                                                                       'source_entry_index': 1,
                                                                       'feedback_behavior': '当前证据支持这一情景时，指出具体问题及希望怎样改；自然生成新表达。'},
                                                                      {'situation': '条件要求餐馆价格在某位旅伴第 3 天午餐 ±$10 内，Agent '
                                                                                    '开场没有点名那位旅伴第 3 '
                                                                                    '天午餐这个参照项，直接去搜索并选了一家。',
                                                                       'user_preference': '遇到跟别人行程项比较的条件，开场先把参照项点名（参照 '
                                                                                          'C2: 某人 第 3 '
                                                                                          '天午餐），查到名称和数值后再补一句。',
                                                                       'user_thinking': '参照项不点名，你比的是哪一顿我都对不上。',
                                                                       'related_rule_ids': ['H3'],
                                                                       'trigger_type': 'correction',
                                                                       'source_entry_index': 2,
                                                                       'feedback_behavior': '当前证据支持这一情景时，指出具体问题及希望怎样改；自然生成新表达。'},
                                                                      {'situation': 'Agent 的开场三行和过程说明全部用英文写。',
                                                                       'user_preference': '说明用中文写，行程正文保持英文格式就行。',
                                                                       'user_thinking': '说明是给我看的，我习惯看中文；行程格式是任务要求的，别动。',
                                                                       'related_rule_ids': ['H6'],
                                                                       'trigger_type': 'correction',
                                                                       'source_entry_index': 3,
                                                                       'feedback_behavior': '当前证据支持这一情景时，指出具体问题及希望怎样改；自然生成新表达。'},
                                                                      {'situation': 'Agent 第一条回复是中文的「目标: / 方法: / '
                                                                                    '拿不准:」三行，接着列出 '
                                                                                    'C1–C4，点名两个参照项，最后给出英文格式行程。',
                                                                       'user_preference': '习惯都做到了，不用纠正。',
                                                                       'user_thinking': '这样我一眼能核对，继续就行。',
                                                                       'related_rule_ids': [],
                                                                       'trigger_type': 'silent_progression',
                                                                       'source_entry_index': 4,
                                                                       'feedback_behavior': '适用要求已满足，不纠正；自然继续当前任务。'},
                                                                      {'situation': '任务只有一条餐馆条件，没有与他人比较；Agent '
                                                                                    '写了中文三行开场后给出行程。',
                                                                       'user_preference': '不适用的习惯（编号、参照点名）不需要出现，无需纠正。',
                                                                       'user_thinking': '条件只有一条，编号和参照值本来就用不上。',
                                                                       'related_rule_ids': [],
                                                                       'trigger_type': 'silent_progression',
                                                                       'source_entry_index': 5,
                                                                       'feedback_behavior': '适用要求已满足，不纠正；自然继续当前任务。'}]},
                                  'style_reference': [{'source_entry_index': 0,
                                                       'kind': 'correction',
                                                       'agent_behavior': '第一条回复直接开始搜索，没有三行开场。',
                                                       'user_response': '先写「目标:」「方法:」「拿不准:」三行再查。',
                                                       'is_paraphrase': True},
                                                      {'source_entry_index': 3,
                                                       'kind': 'correction',
                                                       'agent_behavior': '开场三行和说明全是英文。',
                                                       'user_response': '说明用中文写，行程保持英文格式。',
                                                       'is_paraphrase': True},
                                                      {'source_entry_index': 4,
                                                       'kind': 'silent_progression',
                                                       'agent_behavior': '三行开场、编号、参照值、中文说明都齐了。',
                                                       'user_response': None,
                                                       'is_paraphrase': True}],
                                  'current_cross_task_adaptations': [],
                                  'transfer_examples': [{'example_id': 'transfer-h1-h2',
                                                         'source_rule_ids': ['H1', 'H2'],
                                                         'task': '按几条条件替换一位旅伴行程里的两顿饭和一晚住宿。',
                                                         'correction_case': {'agent_behavior': '第一条回复就调用搜索，没有三行开场，也没有把条件编号列出。',
                                                                             'user_response': '先三行开场，再把条件编成 C1、C2 '
                                                                                              '列出来，然后再查。'},
                                                         'no_correction_case': {'agent_behavior': '第一条回复是中文的三行开场，接着 '
                                                                                                  'C1–C3 '
                                                                                                  '逐条列出条件，之后才开始查询。',
                                                                                'user_response': None}},
                                                        {'example_id': 'transfer-h3',
                                                         'source_rule_ids': ['H3'],
                                                         'task': '条件要求住宿评分不低于同行某人第 2 天住宿，且晚餐价格比其第 1 天晚餐便宜。',
                                                         'correction_case': {'agent_behavior': '开场没有点名那位旅伴第 2 天住宿和第 1 '
                                                                                               '天晚餐这两个参照项就去搜索。',
                                                                             'user_response': '开场先把参照项点名（参照 C1: 某人 第 2 '
                                                                                              '天住宿；参照 C2: 某人 第 1 '
                                                                                              '天晚餐），查到后补数值。'},
                                                         'no_correction_case': {'agent_behavior': '开场写「参照 C1: 某人 第 2 '
                                                                                                  '天住宿」「参照 C2: 某人 第 1 '
                                                                                                  '天晚餐」，查到后补「参照 C1 = '
                                                                                                  '名称, $价格, 评分」。',
                                                                                'user_response': None}},
                                                        {'example_id': 'transfer-h6',
                                                         'source_rule_ids': ['H6'],
                                                         'task': '为某天找一家符合菜系和价格的餐馆。',
                                                         'correction_case': {'agent_behavior': '开场和说明用英文写。',
                                                                             'user_response': '说明用中文，行程保持英文格式。'},
                                                         'no_correction_case': {'agent_behavior': '中文三行开场和说明，行程保持英文格式。',
                                                                                'user_response': None}}],
                                  'interpretation_limits': '这些是我的个人工作习惯，不是当前任务的事实，也不是 agent '
                                                           '已犯错的证据。当前公开任务决定实际要解决的问题；保留各习惯的适用条件和例外，任务明确的输出格式优先。H1/H6 '
                                                           '的"第一条回复"指实际首条公开可见回复；没有首条回复记录时不凭缺失指责。纠正与不纠正情景都参与判断；每个指责要有当前公开证据。纠正采用自然的新表达；以下画像不包含历史用户原句，不能假装旧示例就是这次用户说过的话。'}}}

COOCCURRENCE = {'user_a': {'rule_count': {'H1': 1, 'H2': 1, 'H3': 1, 'H6': 1}, 'cooccur': {}}}
