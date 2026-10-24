"""Optional operator-owned shared API budget, separate from experimental memory."""
import fcntl
import json
import math
import os
from pathlib import Path
import uuid


class SpendingLimitReached(RuntimeError):
    pass


class SpendingBudget:
    def __init__(self, policy):
        self.policy = Path(policy)

    @classmethod
    def for_request(cls, source, identity, model):
        path = getattr(source, 'path', None)
        if path is None or getattr(source, 'provider', None) != 'openrouter':
            return None
        policy = path.with_name('experiment-spending.json')
        if not policy.exists():
            return None
        obj = cls(policy)
        data = obj._policy()
        if not identity.run_id.startswith(data['run_prefix']):
            return None
        admitted = [data['model'], *data.get('additional_models', [])]
        # Separate opt-in leaves the legacy policy readable by already-running
        # frozen workers; both versions continue using the SAME locked ledger.
        extension = policy.with_name('experiment-spending-models.json')
        if model not in admitted and extension.exists():
            extra = json.loads(extension.read_text())
            if (set(extra) != {'schema','primary_model','additional_models'} or extra['schema'] != 1 or
                    extra['primary_model'] != data['model'] or not isinstance(extra['additional_models'],list) or
                    any(not isinstance(m,str) or not m for m in extra['additional_models'])):
                raise SpendingLimitReached('invalid_additional_model_policy')
            admitted.extend(extra['additional_models'])
        if model not in admitted:
            raise SpendingLimitReached('budget_model_not_admitted')
        return obj

    def _policy(self):
        data = json.loads(self.policy.read_text())
        if (set(data)-{'additional_models'} != {'schema','run_prefix','model','ledger','limit_usd','reserve_usd'} or
                data['schema'] != 1 or not isinstance(data['run_prefix'], str) or not data['run_prefix'] or
                not isinstance(data['model'], str) or not data['model'] or
                not isinstance(data.get('additional_models', []), list) or
                any(not isinstance(m,str) or not m for m in data.get('additional_models', [])) or
                not Path(data['ledger']).is_absolute() or
                any(type(data[k]) not in (int,float) or not math.isfinite(data[k]) or data[k] <= 0
                    for k in ('limit_usd','reserve_usd'))):
            raise SpendingLimitReached('invalid_spending_policy')
        return data

    def monitor_only(self):
        """Explicit operator opt-in: retain accounting without a local limit."""
        path = self.policy.with_name('experiment-spending-monitor-only.json')
        if not path.exists():
            return False
        mode = json.loads(path.read_text())
        policy = self._policy()
        if (set(mode) != {'schema', 'mode', 'run_prefix', 'primary_model'} or
                mode['schema'] != 1 or mode['mode'] != 'monitor_only' or
                mode['run_prefix'] != policy['run_prefix'] or
                mode['primary_model'] != policy['model']):
            raise SpendingLimitReached('invalid_monitor_only_policy')
        return True

    def _change(self, operation):
        config = self._policy()
        path = Path(config['ledger'])
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(path.read_text()) if path.exists() else {'spent_usd':0.,'reservations':{},'settled':{}}
            result = operation(data, config)
            temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.new')
            with temp.open('x') as stream:
                json.dump(data, stream, allow_nan=False);stream.flush();os.fsync(stream.fileno())
            os.replace(temp, path)
            return result

    def reserve(self, namespace):
        def update(data, config):
            reserved = sum(data['reservations'].values())
            if not self.monitor_only() and data['spent_usd'] + reserved + config['reserve_usd'] > config['limit_usd']:
                raise SpendingLimitReached('experiment_spending_limit')
            token = namespace + ':' + uuid.uuid4().hex
            data['reservations'][token] = config['reserve_usd']
            return token
        return self._change(update)

    def settle(self, token, cost):
        # Unknown outcomes keep their reservation: crashes cannot refund a
        # possibly charged request. Idempotent receipts never double-charge.
        if cost is None:
            return
        if type(cost) not in (int,float) or not math.isfinite(cost) or cost < 0:
            raise SpendingLimitReached('invalid_reported_charge')
        def update(data, config):
            if token in data['settled']:
                if data['settled'][token] != cost:
                    raise SpendingLimitReached('conflicting_reported_charge')
                return
            reserve = data['reservations'].pop(token)
            data['spent_usd'] += cost
            data['settled'][token] = cost
            # A violated reservation bound halts subsequent dispatches.
            if cost > reserve and not self.monitor_only():
                data['reservations']['reservation_bound_violation'] = config['limit_usd']
        self._change(update)
