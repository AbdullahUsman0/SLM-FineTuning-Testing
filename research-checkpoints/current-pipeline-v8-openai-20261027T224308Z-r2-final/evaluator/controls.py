"""Durable scientific journals and one shared non-renewing hosted budget."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import sqlite3
from pathlib import Path

MODEL = 'gpt-5-mini-2025-08-07'
INPUT_NANOUSD = 250
OUTPUT_NANOUSD = 2000
CEILING_NANOUSD = 900_000_000


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(dumps(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError('Frozen context/configuration drift: ' + path.name)
    else:
        atomic(path, value)


def strict_json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate JSON field')
            obj[key] = value
        return obj
    def nonfinite(value):
        raise ValueError('Nonfinite JSON')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def append(path, value):
    with Path(path).open('ab', buffering=0) as stream:
        stream.write((dumps(value) + '\n').encode())
        os.fsync(stream.fileno())


def read_journal(path):
    path = Path(path)
    if not path.exists():
        return {}
    parts = path.read_bytes().splitlines(keepends=True)
    records = {}
    good = []
    for i, line in enumerate(parts):
        try:
            row = strict_json(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(parts) - 1:
                raise ValueError('Corrupt interior journal')
            backup = path.with_name(path.name + '.incomplete-' + hashlib.sha256(line).hexdigest()[:12])
            if not backup.exists():
                backup.write_bytes(line)
            with path.open('wb') as stream:
                stream.write(b''.join(good)); stream.flush(); os.fsync(stream.fileno())
            break
        key = tuple(row['key'])
        if key in records:
            raise ValueError('Duplicate journal key')
        records[key] = row
        good.append(line if line.endswith(b'\n') else line + b'\n')
    if parts and records and not parts[-1].endswith(b'\n') and len(good) == len(parts):
        with path.open('wb') as stream:
            stream.write(b''.join(good)); stream.flush(); os.fsync(stream.fileno())
    return records


@contextlib.contextmanager
def lock(path):
    """Windows OS lock, released by the OS on process death."""
    import msvcrt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class BudgetStop(RuntimeError):
    pass


class Ledger:
    """SQLite FULL-sync reservations; a pending/uncertain request blocks dispatch."""
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS authorization (id TEXT PRIMARY KEY, config TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, context TEXT NOT NULL, bound INTEGER NOT NULL, actual INTEGER, state TEXT NOT NULL, response_sha TEXT, usage TEXT)')
        expected = dumps({'scope': 'one bounded current-pipeline v8/OpenAI evaluation, including pilots and retries', 'user_ceiling_nanousd': 1_000_000_000, 'dispatch_ceiling_nanousd': CEILING_NANOUSD, 'model': MODEL, 'input_rate_nanousd_per_token': INPUT_NANOUSD, 'output_rate_nanousd_per_token': OUTPUT_NANOUSD})
        self.db.execute('INSERT OR IGNORE INTO authorization VALUES (?, ?)', ('20261023', expected))
        if self.db.execute('SELECT config FROM authorization WHERE id=?', ('20261023',)).fetchone()[0] != expected:
            raise BudgetStop('Authorization/rate drift; budget never resets')

    def summary(self):
        rows = self.db.execute('SELECT key,context,bound,actual,state,response_sha,usage FROM calls').fetchall()
        actual = sum(r[3] or 0 for r in rows)
        reserved = sum(r[2] for r in rows if r[4] != 'reconciled')
        return {'authorization_total_USD': 1.0, 'dispatch_ceiling_USD': 0.9, 'actual_nanousd': actual, 'outstanding_nanousd': reserved, 'actual_USD': actual / 1e9, 'reserved_or_uncertain_USD': reserved / 1e9, 'remaining_dispatch_USD': (CEILING_NANOUSD - actual - reserved) / 1e9, 'calls': [dict(zip(('key','context','bound_nanousd','actual_nanousd','state','response_sha256','usage'), r)) for r in rows]}

    def reserve(self, key, context, input_bound, output_bound=2048):
        if not isinstance(input_bound, int) or input_bound < 1 or output_bound != 2048:
            raise BudgetStop('Invalid conservative token bound')
        bound = input_bound * INPUT_NANOUSD + output_bound * OUTPUT_NANOUSD
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM calls WHERE key=?', (key,)).fetchone():
                raise BudgetStop('Existing dispatch key; do not replay')
            rows = self.db.execute('SELECT bound,actual,state FROM calls').fetchall()
            if any(r[2] != 'reconciled' for r in rows):
                raise BudgetStop('Pending/uncertain request requires reconciliation; no retry')
            if sum(r[1] for r in rows) + bound > CEILING_NANOUSD:
                raise BudgetStop('Internal US$0.90 dispatch ceiling reached')
            self.db.execute('INSERT INTO calls (key,context,bound,state) VALUES (?,?,?,?)', (key,context,bound,'pending'))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK'); raise
        return bound

    def uncertain(self, key):
        self.db.execute("UPDATE calls SET state='uncertain' WHERE key=? AND state='pending'", (key,))

    def reconcile(self, key, context, usage, response_sha):
        # output_tokens already INCLUDES reasoning; never subtract or double-add it.
        input_tokens, output_tokens = usage.get('input_tokens'), usage.get('output_tokens')
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            raise BudgetStop('Missing/incoherent usage; reservation retained')
        reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        if not isinstance(reasoning, int) or reasoning > output_tokens or reasoning < 0:
            raise BudgetStop('Incoherent reasoning usage')
        actual = input_tokens * INPUT_NANOUSD + output_tokens * OUTPUT_NANOUSD
        row = self.db.execute('SELECT context,bound,state,response_sha FROM calls WHERE key=?', (key,)).fetchone()
        if row is None or row[0] != context:
            raise BudgetStop('Unknown dispatch or context drift')
        if row[2] == 'reconciled':
            if row[3] != response_sha:
                raise BudgetStop('Different response for completed request')
            return
        self.db.execute('UPDATE calls SET actual=?,state=?,response_sha=?,usage=? WHERE key=?', (actual,'reconciled',response_sha,dumps(usage),key))
        if actual > row[1] or self.summary()['actual_nanousd'] > CEILING_NANOUSD:
            self.db.execute("UPDATE calls SET state='estimate_exceeded' WHERE key=?", (key,))
            raise BudgetStop('Observed usage exceeded reservation/ceiling; stop')


def request_input_bound(request):
    """Each UTF-8 byte is an upper bound for visible BPE tokens, plus 16K overhead.

    Includes schema and full serialized request, deliberately no cache discount.
    This is an application guard; undocumented provider overhead cannot be guaranteed.
    Reconciled usage exceeding this estimate stops the experiment immediately.
    """
    return len(dumps(request).encode('utf-8')) + 16384
