"""Versioned simulated-user personas (the May simulator layers)."""
from copy import deepcopy
import json

from .may_profile_data import MAY_PREFIX_VOICE, MAY_PROFILES, MAY_PROFILE_VERSION


PROFILE_VERSION = MAY_PROFILE_VERSION
MAX_PROFILE_BYTES = 128 * 1024


def compact_profile(user_id, version=None):
    """Return a fresh copy of a pinned profile, defaulting to the restored May one."""
    version = PROFILE_VERSION if version is None else version
    if version == PROFILE_VERSION:
        profiles = MAY_PROFILES
    else:
        raise ValueError('unknown registered compact/May profile version')
    if not isinstance(user_id, str) or user_id not in profiles:
        raise ValueError('unknown registered compact/May persona')
    return deepcopy(profiles[user_id])


def validate_profile_payload(payload, *, user_id=None, encoded_bytes=None):
    """Accept only exact registered entities; arbitrary source records are forbidden."""
    if not isinstance(payload, dict) or not isinstance(payload.get('kind'), str):
        raise ValueError('profile must match a fixed compact/May persona')
    owner = payload.get('user_id')
    if user_id is not None and owner != user_id:
        raise ValueError('profile owner does not match its role identity')
    if payload != compact_profile(owner, version=payload['kind']):
        raise ValueError('profile must match a fixed compact/May persona; raw records are forbidden')
    limit = MAX_PROFILE_BYTES
    actual_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    if actual_bytes > limit or (encoded_bytes is not None and encoded_bytes > limit):
        raise ValueError('registered profile exceeds its versioned byte limit')
    return payload


def feedback_profile_projection(payload):
    """Send May's full simulator layers once; the description is a judge projection."""
    validate_profile_payload(payload)
    if payload['kind'] == PROFILE_VERSION:
        keys = ('kind', 'user_id', 'source_id', 'simulator_profile')
    else:
        keys = ('kind', 'user_id', 'source_id', 'description')
    return {key: deepcopy(payload[key]) for key in keys}


def render_may_prefix(profile):
    """Render the frozen simulator layers without Git, documents or source I/O."""
    if (not isinstance(profile, dict) or not isinstance(profile.get('user_id'), str)
            or profile['user_id'] not in MAY_PROFILES):
        raise ValueError('a registered May simulator profile is required')
    if profile != MAY_PROFILES[profile['user_id']]['simulator_profile']:
        raise ValueError('May simulator profile differs from its frozen source entity')
    parts = [MAY_PREFIX_VOICE, '# Profile\n## Layer 1 - 项目目的',
             json.dumps(profile['layer_1_projects'], ensure_ascii=False),
             '## Layer 2 - 完整 May 偏好及其适用边界',
             json.dumps(profile['layer_2_preferences'], ensure_ascii=False),
             '## Layer 3 - May Conditional entries（纠正与不纠正均保留）']
    for entry in profile['layer_3']['conditional_entries']:
        parts.append(f"\n### 条目 {entry['source_entry_index']} [{entry['trigger_type']}]\n"
                     f"情景: {entry.get('situation', '')}\n偏好: {entry.get('user_preference', '')}\n"
                     f"心想: {entry.get('user_thinking', '')}\n反馈行为: {entry['feedback_behavior']}\n"
                     f"相关规则: {entry.get('related_rule_ids', [])}")
    parts += ['# Few-shot 参考风格：1 个纠正、2 个无需纠正的释义例，不是历史用户原话',
              json.dumps(profile['style_reference'], ensure_ascii=False),
              '# 用户本轮明确认可的少量跨任务适配（与 May 原字段分列）',
              json.dumps(profile['current_cross_task_adaptations'], ensure_ascii=False),
              '# 跨任务纠正/不纠正成对参考（规则抽象，不是历史实录；不改变原 Layer 3 索引）',
              json.dumps(profile['transfer_examples'], ensure_ascii=False),
              '# 当前证据、可观察性与授权边界', profile['interpretation_limits']]
    return '\n'.join(parts)
