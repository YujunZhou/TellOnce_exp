"""Pinned upstream travel output parsing, preserving latest plans per person.

The task owner supplies only delivered texts captured by the scoring boundary.
Pure upstream functions are loaded without importing their dataset, models,
memory clients or optional dependencies. This module cannot start an agent.
"""
import ast
import hashlib
from pathlib import Path
import re

from tellonce_common.role_inputs import PublicUnit


PINNED = {
    'run_travel.py': ('3b35937f0caba4c2b614f42d1b0cd0d4f32242973e4942bca5eda89df82e5d5f', 'parse_all_plans'),
    'env/env_systems/travel_planner_env/combination.py':
        ('e69ee0eef84ee6941d833a77db467179c37d61a7b544c4dc77b011f54d22fdae', 'parse_plan_text'),
    'env/env_systems/travel_planner_env/data_loader.py':
        ('8033cc39ab6514f9e79a7044ab84b4dffd1f6a72dcbe4a6281ea49766076c7af', '_extract_name'),
    'env/env_systems/travel_planner_env/prompts.py':
        ('984d2c9dc8522ec750eb8ed4da41eb2397997ba80272faac918ba80fe7149ade', 'AGENT_SYSTEM_PROMPT'),
}


from tellonce_common.public_task_contracts import GROUP_PLAN_CONTRACT

TRAVEL_PUBLIC_INPUT_VERSION = 'group-travel-mode-preserving-format-v2'


def original_output_instructions(source: Path) -> str:
    relative='env/env_systems/travel_planner_env/prompts.py'
    path=source/relative
    if not path.is_file() or path.is_symlink():
        raise ValueError('original public travel prompt source changed or is missing')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINNED[relative][0]:
        raise ValueError('original public travel prompt source changed or is missing')
    nodes=[node for node in ast.parse(raw,filename=str(path)).body
           if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
           and node.targets[0].id=='AGENT_SYSTEM_PROMPT']
    if len(nodes)!=1:
        raise ValueError('original public travel output instructions are missing')
    return ast.literal_eval(nodes[0].value)


def adapted_output_instructions(source: Path) -> str:
    """Resolve the public flight-only template conflict for newly built tasks.

    The upstream parser accepts a transportation string for any travel mode.
    Keep its headers/field names and other output constraints, but do not let
    an example flight format override an inherited confirmed driving route.
    The pinned original prompt remains available for old captured experiments.
    """
    text = original_output_instructions(source)
    replacements = {
        '1. First, use the tools to search for available flights, restaurants, accommodations, and attractions':
            '1. Use the tools to check the new or changed selections and constraints requested by the '
            'current traveler. Preserve confirmed inherited fields unless the current request or its '
            'necessary consequences change them. Search flights when a flight is requested or needed '
            'for an authorized transport change; use DistanceMatrix for driving changes when needed.',
        'Transportation: Flight Number: {flight_number}, from {origin} to {destination}, Departure Time: {dep_time}, Arrival Time: {arr_time}':
            'Transportation: {transportation_required_for_this_day}',
        'Transportation: Flight Number: {flight_number}, from {destination} to {origin}, Departure Time: {dep_time}, Arrival Time: {arr_time}':
            'Transportation: {transportation_required_for_this_day}',
        'Transportation: -': 'Transportation: {transportation_required_for_this_day}',
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise ValueError('pinned travel template changed; review the mode-preserving adaptation')
        text = text.replace(before, after, 1)
    return text + '''
**TRANSPORTATION VALUES:**
- The Transportation field records the actual mode required by the current traveler's requests and the inherited confirmed plan. The template does not require switching to a flight.
- If no transport change is requested or required by another explicit change, preserve the confirmed Transportation value exactly, including a self-driving route.
- For a flight, use: Flight Number: {flight_number}, from {origin} to {destination}, Departure Time: {dep_time}, Arrival Time: {arr_time}.
- For a new or changed driving route, use: Self-driving, from {origin} to {destination}, duration: {duration}, distance: {distance}, cost: {cost}.
- Use "-" only when transportation is unnecessary. Fill the placeholder with the actual value; do not output placeholder braces. The no-extra-prices rule prohibits added commentary, not required details inside an inherited or mode-specific Transportation value.
'''


class NativeTravelSubmission:
    def __init__(self, source: Path, *, units: tuple[PublicUnit, ...]):
        if (not isinstance(units,tuple) or not units or any(not isinstance(u,PublicUnit) for u in units) or
                any(u.instance.benchmark!='memoryarena' or u.instance.task_family!='group_travel_planner' or
                    u.independent_task_id!=units[0].independent_task_id or
                    u.instance.profile_id!=units[0].instance.profile_id or
                    u.instance.dataset_revision!=units[0].instance.dataset_revision or
                    u.instance.split!=units[0].instance.split or u.instance.subtask_id!=str(i+1)
                    for i,u in enumerate(units)) or
                len({u.instance.common_instance_id for u in units})!=len(units)):
            raise ValueError('complete ordered public travel task required')
        if not isinstance(source,Path) or not source.is_absolute() or source!=source.resolve():
            raise ValueError('pinned original travel source directory required')
        self._native={'re':re}
        for relative,(digest,name) in PINNED.items():
            path=source/relative
            if not path.is_file() or path.is_symlink():
                raise ValueError('pinned travel source is missing')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:
                raise ValueError('original travel source changed after review')
            tree=ast.parse(raw,filename=str(path))
            if name=='AGENT_SYSTEM_PROMPT':
                nodes=[node for node in tree.body if isinstance(node,ast.Assign) and
                       len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name]
                if len(nodes)!=1:
                    raise ValueError('pinned original output instructions are missing')
                self.output_instructions=ast.literal_eval(nodes[0].value)
            else:
                nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
                if len(nodes)!=1:
                    raise ValueError('pinned original parsing function is missing')
                exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),self._native)
        self._unit_ids=tuple(u.instance.common_instance_id for u in units)
        self._queries=tuple({'name':self._native['_extract_name'](u.instruction),'query':u.instruction} for u in units)

    def plans_at(self, unit_id, *, delivered_texts: tuple[str,...]):
        if (unit_id not in self._unit_ids or not isinstance(delivered_texts,tuple) or
                any(not isinstance(text,str) for text in delivered_texts)):
            raise ValueError('actual scoring unit and immutable delivered text history required')
        index=self._unit_ids.index(unit_id)
        # Match upstream accumulated_plan_text and parse_all_plans: the latest
        # named section wins, and an omitted person retains their earlier plan.
        # Only travelers introduced by this boundary contribute to the result.
        rows=self._native['parse_all_plans']('\n\n'.join(delivered_texts),self._queries[:index+1])
        return tuple({**row,'plan':self._native['parse_plan_text'](row['result'])} for row in rows)
