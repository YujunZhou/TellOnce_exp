"""Task-owned JSON-RPC stdio service for the unchanged public MemoryArena tools.

Run inside the native filesystem/network sandbox with an empty CLI home. This
service has no model, credentials, task answer file, or TravelEnvironment.reset.
The host owns deadlines, process lifetime and the current native call identity.
"""
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time


TOOL_ARGUMENTS = {
    'FlightSearch': ('origin', 'destination', 'date'),
    'RestaurantSearch': ('city',), 'AccommodationSearch': ('city',),
    'AttractionSearch': ('city',), 'DistanceMatrix': ('origin', 'destination', 'mode'),
    'CitySearch': ('state',),
}
MAX_REQUEST_CHARACTERS = 65536
MAX_OUTPUT_BYTES = 2*1024*1024


def native_tool_definitions():
    """Expose original public query signatures, without reset or answer fields."""
    city_lookup = (
        'City lookup uses the exact city name stored in the database. A plan label '
        'may append a state in parentheses or after a comma; that full label may '
        'not be a database key. If a search returns no rows, use CitySearch for '
        'the stated state and retry with the returned city name before concluding '
        'the database has no options. Do not change the itinerary destination.'
    )
    descriptions = {
        'FlightSearch': 'Search the public benchmark flight database by origin, destination and date (YYYY-MM-DD).',
        'RestaurantSearch': 'Search public benchmark restaurants in a city.',
        'AccommodationSearch': 'Search public benchmark accommodation options in a city.',
        'AttractionSearch': 'Search public benchmark attractions in a city.',
        'DistanceMatrix': 'Look up public benchmark travel distance and duration between origin and destination.',
        'CitySearch': 'Search public benchmark cities in a state.',
    }
    result = []
    for name, arguments in TOOL_ARGUMENTS.items():
        properties = {arg: {'type': 'string'} for arg in arguments}
        if 'city' in properties:
            properties['city']['description'] = city_lookup
        if name == 'DistanceMatrix':
            properties['mode']['enum'] = ['driving', 'taxi']
        result.append({'type': 'function', 'name': name, 'description': descriptions[name],
            'inputSchema': {'type': 'object', 'additionalProperties': False,
                            'properties': properties, 'required': list(arguments)}})
    return result


def validate_owner(owner):
    if (not isinstance(owner, dict) or set(owner) != {'namespace', 'task_id', 'session_id'}
            or any(not isinstance(v, str) or not 1 <= len(v) <= 480 for v in owner.values())
            or not owner['namespace'].startswith('experiment_')
            or not owner['task_id'].startswith('group_travel_planner:')):
        raise ValueError('explicit experiment task and session required')


def validate_request(value, owner):
    validate_owner(owner)
    if (not isinstance(value, dict) or set(value) != {'context', 'tool', 'args'}
            or not isinstance(value['context'], dict)
            or set(value['context']) != set(owner)|{'unit_id', 'event_id', 'call_id'}
            or any(value['context'][k] != v for k, v in owner.items())
            or any(not isinstance(v, str) or not 1 <= len(v) <= 480 for v in value['context'].values())
            or not isinstance(value['tool'], str) or value['tool'] not in TOOL_ARGUMENTS
            or not isinstance(value['args'], dict) or set(value['args']) != set(TOOL_ARGUMENTS[value['tool']])
            or any(not isinstance(v, str) or not v.strip() or len(v) > 8192 for v in value['args'].values())
            or (value['tool'] == 'DistanceMatrix' and value['args']['mode'] not in {'driving', 'taxi'})):
        raise ValueError('tool request is malformed, unsupported or belongs to another task')


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON field')
        value[key] = item
    return value


def _constant(value):
    raise ValueError('nonfinite JSON value')


def decode(line):
    return json.loads(line, object_pairs_hook=_object, parse_constant=_constant)


def rpc_request(value, method):
    if (not isinstance(value, dict) or set(value) != {'id', 'method', 'params'}
            or type(value['id']) is not int or value['id'] < 1 or value['method'] != method
            or not isinstance(value['params'], dict)):
        raise ValueError('bounded identified service request required')
    return value['id'], value['params']


def serve(source, destination, *, owner, executor, diagnostics, output_limit=MAX_OUTPUT_BYTES, ready_id=1):
    validate_owner(owner)
    if type(output_limit) is not int or not 1 <= output_limit <= MAX_OUTPUT_BYTES:
        raise ValueError('bounded tool output required')
    def emit(value):
        destination.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        destination.flush()
    if type(ready_id) is not int or ready_id < 1:
        raise ValueError('initialization identity required')
    emit({'id': ready_id, 'result': {'type': 'ready', 'owner': owner}})
    seen, request_ids = set(), {ready_id}
    while True:
        line = source.readline(MAX_REQUEST_CHARACTERS+1)
        if not line:
            return
        if len(line) > MAX_REQUEST_CHARACTERS or not line.endswith('\n'):
            emit({'method': 'service/protocol_error', 'params': {'error_code': 'request_budget_or_incomplete'}})
            return  # Never execute a prefix or reinterpret the tail as another request.
        try:
            identity, value = rpc_request(decode(line), 'tool/call')
            if identity in request_ids:
                raise ValueError('duplicate RPC identity')
            request_ids.add(identity)
        except (ValueError, TypeError):
            emit({'method': 'service/protocol_error', 'params': {'error_code': 'invalid_rpc'}})
            return
        try:
            validate_request(value, owner)
            call_id = value['context']['call_id']
            if call_id in seen:
                raise ValueError('duplicate call identity')
        except (ValueError, TypeError):
            emit({'id': identity, 'result': {'type': 'tool_result', 'owner': owner, 'status': 'invalid_request'}})
            continue
        seen.add(call_id)
        started = time.monotonic()
        reply = {'type': 'tool_result', 'owner': owner, 'context': value['context']}
        try:
            # Official tools log diagnostics with print(). Keep protocol stdout
            # separate while preserving the exact string returned by execute().
            with redirect_stdout(diagnostics):
                result = executor.execute(value['tool'], value['args'])
            if not isinstance(result, str):
                raise TypeError('original tool must return text')
            data = result.encode('utf-8')
            if len(data) > output_limit:
                reply.update(status='unavailable', error_code='output_budget')
            else:
                reply.update(status='completed', result=result, result_sha256=hashlib.sha256(data).hexdigest())
        except Exception as exc:
            reply.update(status='unavailable', error_code='native_tool_exception', error_type=type(exc).__name__)
        reply['wall_s'] = time.monotonic()-started
        envelope = {'id': identity, 'result': reply}
        if len(json.dumps(envelope, ensure_ascii=False).encode()) > MAX_OUTPUT_BYTES:
            reply.pop('result', None)
            reply.pop('result_sha256', None)
            reply.update(status='unavailable', error_code='wire_output_budget')
        emit(envelope)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_tree(root, expected):
    if not isinstance(root, Path) or not root.is_absolute() or root != root.resolve() or not root.is_dir():
        raise ValueError('canonical public resource directory required')
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError('public resources cannot contain symlinks or special files')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    if set(files) != set(expected):
        raise ValueError('public resource tree differs from its fixed manifest')
    for relative, path in files.items():
        if _hash(path) != expected[relative]:
            raise ValueError('original public resource changed')


def main():
    line = sys.stdin.readline(MAX_REQUEST_CHARACTERS+1)
    if len(line) > MAX_REQUEST_CHARACTERS or not line.endswith('\n'):
        raise ValueError('bounded complete initialization required')
    ready_id, setup = rpc_request(decode(line), 'initialize')
    if not isinstance(setup, dict) or set(setup) != {'owner', 'source_root', 'database_root'}:
        raise ValueError('public-resource initialization required')
    validate_owner(setup['owner'])
    pins = decode(Path(__file__).with_name('travel_tool_pins.json').read_text())
    source, database = Path(setup['source_root']), Path(setup['database_root'])
    verify_public_tree(source, pins['sources'])
    verify_public_tree(database, pins['database'])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with redirect_stdout(sys.stderr):
        from travel_flight_index import public_executor
        executor = public_executor(database, source_sha256=pins['database']['flights/clean_Flights_2022.csv'])
    try:
        serve(sys.stdin, sys.stdout, owner=setup['owner'], executor=executor, diagnostics=sys.stderr, ready_id=ready_id)
    finally:
        executor.flights.close()


if __name__ == '__main__':
    main()
