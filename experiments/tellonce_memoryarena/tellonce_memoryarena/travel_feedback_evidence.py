"""Read named-venue facts from public tool CSVs; never read task answer files."""
import csv
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

SOURCES = (('restaurant','restaurants/clean_restaurant_2022.csv','Name'),
           ('accommodation','accommodations/clean_accommodations_2022.csv','NAME'))

_CHOICES = {'Current City', 'Breakfast', 'Lunch', 'Dinner', 'Accommodation'}
_ORDINALS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
             'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
             'ninth': 9, 'tenth': 10}


def _normalized(text):
    text=' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    text=text.translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','–':'-','—':'-'}))
    return re.sub(r'\s*([,;:/-])\s*',r'\1',text)


def _mentioned(name, text):
    return bool(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text))


def _city_key(text):
    # Public plans add region qualifiers (e.g. Toledo(Ohio)) while venue CSVs
    # store only the city. This lookup verifies the city, not the qualifier.
    return re.sub(r'\s*\([^()]*\)\s*$', '', _normalized(text)).strip()


def _requirements(instruction):
    # Remove only whole introductory sentences, not constraints that follow them.
    return re.sub(r"(?im)^\s*(?:i am [\w -]+|(?:i'm|i am|we're|we are) "
                  r"(?:going on this trip with|travell?ing with|joining)\b[^.\n]*)[.!]?(?:\s*\n|$)",
                  '', instruction)


def _relevant_days(record, requirements):
    """Project a safe superset of literal referenced slots, without inferring facts.

    Both the subject day and comparison day in a sentence are retained. If a
    reference has no explicit day or slot, keep the entire plan as a fallback.
    """
    lines = [s for s in requirements.splitlines() if _mentioned(record['person'].casefold(), s)]
    if not lines:
        return record
    wanted = {}
    for line in lines:
        days = {int(n) for n in re.findall(r'\bday\s+(\d+)\b', line)}
        days.update(int(n) for n in re.findall(r'\b(\d+)(?:st|nd|rd|th)?[- ]day\b', line))
        days.update(_ORDINALS[n] for n in re.findall(
            r'\b(' + '|'.join(_ORDINALS) + r')[- ]day\b', line))
        fields = {k for k in _CHOICES if k != 'Current City' and _mentioned(k.casefold(), line)}
        if re.search(r'\b(?:stay|lodging|room)\b', line):
            fields.add('Accommodation')
        if not days or not fields:
            return record
        for day in days:
            wanted.setdefault(day, set()).update(fields | {'Current City'})
    selected = [{'day': d['day'], 'choices': {k: v for k, v in d['choices'].items()
                 if k in wanted[d['day']]}} for d in record['days'] if d['day'] in wanted]
    return {**record, 'days': selected}

@lru_cache(maxsize=4)
def _rows(path, stamp, size):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return tuple({k:v for k,v in row.items() if k and not k.startswith('Unnamed:')} for row in csv.DictReader(f))

def execution_history_context(public_task, history, *, user_text, delivery=None):
    """Literal, task-local reference choices for execution checks, without I/O."""
    task=json.loads(public_task)
    if (not isinstance(task,dict) or not isinstance(history,(list,tuple)) or
            any(not isinstance(row,dict) or row.get('role') not in {'user','assistant'} or
                not isinstance(row.get('text'),str) for row in history)):
        raise ValueError('attributed public task history required')
    instruction=task['instruction']
    # A trip introduction lists companions, not comparison targets. Supplying
    # all their plans adds unrelated constraints and can obscure the requested
    # person/day. Retain only names used by actual requirements or corrections.
    payload={'instruction':instruction,'visible_history':history}
    # Local corrections can refer to another person absent from the original
    # question. Both texts are public; neither is replaced by a summary.
    if user_text!=instruction:
        payload['instruction']=instruction+'\n'+user_text
    base=task.get('context',{}).get('base_person',{})
    return {'initial_plan_owner':base.get('name'),
            'referenced_prior_choices':referenced_prior_choices(
                payload, requirements_only=True, current_user_text=user_text),
            'interpretation':'Names identify different people. The initial plan belongs only to its named owner. '
                'For an explicit reference to another person, use that person\'s latest captured choices, '
                'with later local edits changing only their stated slots. Do not substitute the initial owner, '
                'another person, or another day. Missing reference choices are unknown, not a mismatch. '
                'Bind each comparison to exactly the properties and quantifiers requested: sharing a named '
                'property with another choice does not require sharing every property of that choice. '
                'For example, a named cuisine "like" another meal requires that named cuisine; '
                '"all the same cuisines" requires the full reference set, and "only" additionally excludes extras. '
                'Reference values identify what to compare, not additional requirements. '
                'These are literal prior deliveries, not verified correctness or reference answers. '
                'The complete current candidate is observation.delivery; read it directly. '
                'This history contains reference choices only, not a replacement or truncation of that candidate.'}


def referenced_prior_choices(payload, *, requirements_only=False, current_user_text=None):
    """Extract latest named plans in this task for people explicitly referenced.

    This is a literal projection of past deliveries, not a generated summary or
    a correctness judgment. Only meal/lodging/city lines are relevant here.
    """
    instruction=payload.get('instruction','').casefold();people={};latest_responses={};owner=None
    reference_text=_requirements(instruction) if requirements_only else instruction
    for turn in payload.get('visible_history',[]):
        if turn.get('role')=='user':
            named=re.match(r"\s*I am ([\w -]+)[.!](?:\s|$)",turn['text'],re.I)
            if named:owner=named[1].strip()
        if turn.get('role')!='assistant':continue
        if owner and _mentioned(owner.casefold(),reference_text):latest_responses[owner]=turn['text']
        for match in re.finditer(r"===\s*([^\n=]+?)['’]s Plan\s*===\s*\n(.*?)(?====|\Z)",turn['text'],re.S):
            names=[n.strip() for n in re.split(r',\s*|\s+and\s+|\s*&\s*',match[1].strip())]
            names=[n for n in names if n and _mentioned(n.casefold(),reference_text)]
            if not names:continue
            days=[];day=None
            for line in match[2].splitlines():
                found=re.fullmatch(r'\s*Day (\d+):\s*',line)
                if found:
                    day={'day':int(found[1]),'choices':{}};days.append(day);continue
                if day is None or ':' not in line:continue
                field,value=line.strip().split(':',1)
                if field in _CHOICES:
                    day['choices'][field]=value.strip()
            for name in names if days else []:
                key=name.casefold()
                prior=people.setdefault(key,{'person':name,'days':[]})
                merged={d['day']:d for d in prior['days']}
                for day in days:
                    merged.setdefault(day['day'],{'day':day['day'],'choices':{}})['choices'].update(day['choices'])
                prior['days']=[merged[n] for n in sorted(merged)]
    # A trip introduction can list everyone. Under the cap, retain the people
    # needed by actual constraints before incidental names in that introduction.
    # Execution can receive the current question separately from prior history.
    # Do not mistake the preceding traveler's question for a local correction.
    if current_user_text is not None and not isinstance(current_user_text,str):
        raise ValueError('current user text must be a string')
    latest=(current_user_text.casefold() if current_user_text is not None else
            next((turn['text'].casefold() for turn in reversed(payload.get('visible_history',[]))
                  if turn.get('role')=='user'), instruction))
    requirements=_requirements(instruction)
    # A later prose clarification can withdraw or qualify an earlier plan.
    # Keep attributed literal excerpts rather than silently trusting stale slots.
    excerpts=[];excerpt_size=0
    for name,text in latest_responses.items():
        if not _mentioned(name.casefold(),requirements):continue
        if re.search(r"===\s*"+re.escape(name)+r"['’]s Plan\s*===",text,re.I):continue
        lines=[line for line in text.splitlines() if line.strip()]
        selected=[];size=0
        for line in lines:
            if size+len(line)>1100:continue
            selected.append(line);size+=len(line)
        row={'person':name,'text':'\n'.join(selected),'complete':len(selected)==len(lines)}
        cost=len(json.dumps(row,ensure_ascii=False))
        if excerpt_size+cost<=2500:excerpts.append(row);excerpt_size+=cost
    def priority(record):
        pattern=r'(?<!\w)'+re.escape(record['person'].casefold())+r'(?!\w)'
        return (int(latest!=instruction and bool(re.search(pattern,latest))),
                int(bool(re.search(pattern,requirements))))
    selected=[];used=excerpt_size;omitted_slots=0
    for full in sorted(people.values(),key=priority,reverse=True):
        later_requirements=_requirements(latest) if requirements_only else latest
        record=_relevant_days(full, requirements + ('\n'+later_requirements if latest!=instruction else ''))
        omitted_slots += sum(len(d['choices']) for d in full['days'])-sum(len(d['choices']) for d in record['days'])
        cost=len(json.dumps(record,ensure_ascii=False))
        if used+cost<=6000:selected.append(record);used+=cost
    return {'records':selected,'omitted_people':len(people)-len(selected),
            'unselected_slots':omitted_slots,
            'later_response_excerpts':excerpts,
            'source':'literal named choices from earlier deliveries in this task; later day/field edits replace only those slots; claims, not verified success'}


def _strings(value):
    """Avoid JSON escaping quotes inside venue names before literal matching."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _named_selections(delivery):
    """Only explicit meal/lodging lines with a stated city; no prose guessing."""
    day=None;person=None
    for line in delivery.splitlines():
        heading=re.fullmatch(r"\s*===\s*([^\n=]+?)['’]s Plan\s*===\s*",line)
        if heading:
            person=heading[1].strip();day=None;continue
        found=re.fullmatch(r'\s*Day (\d+):\s*',line)
        if found:day=int(found[1]);continue
        found=re.fullmatch(r'\s*(Breakfast|Lunch|Dinner|Accommodation):\s*(.+)',line)
        if not found or ',' not in found[2]:continue
        name,city=found[2].rsplit(',',1)
        if not name.strip() or not city.strip():continue
        choice={'day':day,'slot':found[1],'name':name.strip(),'city':city.strip()}
        if person is not None:choice['person']=person
        yield choice


def current_selection_index(delivery, *, max_chars=1800):
    """Literal current-answer locations, not inferred history or compliance."""
    records=[];used=0;omitted=0
    for choice in _named_selections(delivery):
        cost=len(json.dumps(choice,ensure_ascii=False))
        if used+cost>max_chars:
            omitted+=1;continue
        records.append(choice);used+=cost
    return {'source':'literal named selections in CURRENT delivery only',
            'records':records,'omitted':omitted,
            'coverage':'named meal/lodging lines with stated cities; missing slots may be omitted, unset, or prose; read delivery'}


def initial_plan_changes(payload, *, max_chars=2600):
    """Bounded literal differences, not expected answers or violation labels.

    Compare only an explicitly named current traveler against the public base.
    Prior corrections and requested exceptions remain the feedback model's job.
    Never turn an absent field into a deletion or join different people's plans.
    """
    context = payload.get('context', {})
    base = context.get('base_person', {})
    if not context.get('travel_plan_contract') or not isinstance(base, dict):
        return None
    named = re.match(r"\s*I am ([\w -]+)[.!](?:\s|$)", payload.get('instruction', ''), re.I)
    if not named or not isinstance(base.get('daily_plans'), list):
        return None
    person = named[1].strip()
    fields = {'current_city': 'Current City', 'transportation': 'Transportation',
              'breakfast': 'Breakfast', 'lunch': 'Lunch', 'dinner': 'Dinner',
              'attraction': 'Attraction', 'accommodation': 'Accommodation'}
    originals = {}
    for row in base['daily_plans']:
        if not isinstance(row, dict):
            continue
        day = row.get('day', row.get('days'))
        if type(day) is not int:
            continue
        for key, label in fields.items():
            value = row.get(key, row.get(label))
            if isinstance(value, str):
                originals[day, label] = value
    records = []; used = omitted = 0; owner = None; day = None; observed = {}
    for line in payload.get('delivery', '').splitlines():
        heading = re.fullmatch(r"\s*===\s*([^\n=]+?)['’]s Plan\s*===\s*", line)
        if heading:
            owners = re.split(r',\s*|\s+and\s+|\s*&\s*', heading[1].strip())
            owner = person if person.casefold() in {n.strip().casefold() for n in owners} else None
            if owner is not None:
                # Match the latest named submission, not a discarded draft
                # printed earlier in the same response. Do not merge sections.
                observed = {}
            day = None
            continue
        match = re.fullmatch(r'\s*Day (\d+):\s*', line)
        if match:
            day = int(match[1]); continue
        if owner is None or ':' not in line:
            continue
        field, value = line.strip().split(':', 1)
        if field in fields.values():
            observed[day, field] = value.strip()
    for (day, field), after in observed.items():
        before = originals.get((day, field))
        if before is None or before.strip() == after:
            continue
        row = {'person': person, 'day': day, 'field': field, 'initial': before, 'current': after}
        cost = len(json.dumps(row, ensure_ascii=False))
        if used + cost > max_chars:
            omitted += 1
        else:
            records.append(row); used += cost
    return {'base_person': base.get('name'), 'records': records, 'omitted': omitted,
            'source': 'literal differences from the public INITIAL plan, not the latest accepted plan',
            'coverage': 'explicit named day/field lines only; absent fields are not deletions; '
                        'differences may be requested, necessary, previously accepted, or cosmetic; '
                        'use current instructions and earlier corrections before judging'}


def venue_evidence(database, payload):
    """Match names appearing in the current answer, rolling context or public base.

    Duplicate names retain their city and attributes; no verdict is precomputed.
    Whole records only, with explicit count of selected matches omitted by cap.
    """
    context=payload.get('context',{})
    prior=referenced_prior_choices(payload)
    sections = [payload.get('delivery',''), '\n'.join(_strings(prior)),
        '\n'.join(_strings(context.get('base_person',{}))), payload.get('instruction',''),
        '\n'.join(_strings(payload.get('unreviewed_outputs',[]))),payload.get('previous_output_summary','')]
    sections = [_normalized(s) for s in sections]
    text='\n'.join(sections)
    current=current_selection_index(payload.get('delivery',''))
    changes=initial_plan_changes(payload)
    # Share the existing record budget: no larger input window or model call.
    records=[];matched=0;used=len(json.dumps(current,ensure_ascii=False));unavailable=[];candidates=[];source_rows={}
    if changes is not None:
        used += len(json.dumps(changes,ensure_ascii=False))
    for kind,relative,namekey in SOURCES:
        path=Path(database)/relative
        if not path.is_file():
            unavailable.append(relative);continue
        metadata=path.stat()
        source_rows[kind]=_rows(str(path),metadata.st_mtime_ns,metadata.st_size)
        for row in source_rows[kind]:
            name=_normalized(row.get(namekey,'')).rstrip('.!')
            if len(name)<3 or name not in text or not _mentioned(name,text):continue
            # Disambiguate with an explicitly mentioned database city. Names are
            # opaque; do not infer a city from the words inside the venue name.
            city=_normalized(row.get('City',row.get('city','')))
            if city and not _mentioned(city,text):continue
            matched+=1;record={'kind':kind,'attributes':row}
            priority=next(i for i,s in enumerate(sections) if _mentioned(name,s))
            candidates.append((priority,record))
    issues=[];issue_size=0;omitted_issues=0
    for choice in _named_selections(payload.get('delivery','')):
        kind='accommodation' if choice['slot']=='Accommodation' else 'restaurant'
        if kind not in source_rows:continue
        namekey='NAME' if kind=='accommodation' else 'Name'
        key=_normalized(choice['name']).rstrip('.!')
        exact=[row for row in source_rows[kind] if _normalized(row.get(namekey,'')).rstrip('.!')==key]
        cities=sorted({row.get('city',row.get('City','')).strip() for row in exact})
        if any(_city_key(c)==_city_key(choice['city']) for c in cities):continue
        issue={**choice,'lookup':'name_found_in_other_city' if exact else 'no_exact_name_in_public_database',
               'recorded_cities':cities}
        # Similar strings may be abbreviations, but are never silently accepted
        # as the same venue. The model can ask for the exact recorded name.
        cost=len(json.dumps(issue,ensure_ascii=False))
        if issue_size+cost<=2000:issues.append(issue);issue_size+=cost
        else:omitted_issues+=1
        for row in exact:
            record={'kind':kind,'attributes':row}
            if not any(c[1]==record for c in candidates):candidates.append((0,record));matched+=1
    for _, record in sorted(candidates,key=lambda c:c[0]):
        cost=len(json.dumps(record,ensure_ascii=False))
        if used+cost>14000:continue
        records.append(record);used+=cost
    result = {'source':'public benchmark restaurant/accommodation database',
            'current_selections':current,
            'referenced_prior_choices':prior,'unavailable_sources':unavailable,'records':records,'matched_records':matched,'omitted_records':matched-len(records),
            'selection_lookup_issues':issues,'omitted_lookup_issues':omitted_issues,
            'coverage':'named venues and exact-name lookup issues; city region qualifiers not checked; no gold or verdict; other tools not included'}
    if changes is not None:
        result['initial_plan_changes'] = changes
    return result
