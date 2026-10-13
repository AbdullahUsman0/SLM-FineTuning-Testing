"""Checksummed per-call journals and conservative pre-request budget reservation."""
from pathlib import Path
import runtime as rt


class Journal:
    def __init__(self,root,identity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.identity=identity

    def save(self,key,payload):
        rt.atomic(self.root/(key+'.json'),{'identity':self.identity,'payload_sha256':rt.digest(payload),'payload':payload},exclusive=True)

    def load(self,key):
        path=self.root/(key+'.json')
        if not path.exists():return None
        record=rt.read(path)
        if record['identity']!=self.identity or record['payload_sha256']!=rt.digest(record['payload']):
            raise ValueError('Resume journal integrity or identity mismatch')
        return record['payload']


class Budget:
    def __init__(self,path,*,identity,cap_usd,attempt_bound,max_attempts,max_attempts_per_call=2):
        self.path=Path(path);self.identity=identity;self.cap=float(cap_usd)
        self.bound=float(attempt_bound);self.max_attempts=int(max_attempts);self.per_call=max_attempts_per_call
        if not self.path.exists():
            rt.atomic(self.path,{'identity':identity,'approved_cap_usd':self.cap,'attempt_bound_usd':self.bound,
                'max_attempts':self.max_attempts,'max_attempts_per_call':self.per_call,'attempts':[]},exclusive=True)
        self.data=rt.read(self.path)
        if any(self.data.get(key)!=value for key,value in {'identity':identity,'approved_cap_usd':self.cap,
            'attempt_bound_usd':self.bound,'max_attempts':self.max_attempts,'max_attempts_per_call':self.per_call}.items()):
            raise ValueError('Budget identity changed on resume')

    @property
    def exposure(self):
        return sum(a['actual_cost_usd'] if a.get('actual_cost_usd') is not None else a['reserved_usd'] for a in self.data['attempts'])

    def reserve(self,call_id):
        if len(self.data['attempts'])>=self.max_attempts:raise RuntimeError('Registered API attempt allowance exhausted')
        if sum(a['call_id']==call_id for a in self.data['attempts'])>=self.per_call:
            raise RuntimeError('Transport attempt allowance for this call exhausted')
        if self.exposure+self.bound>self.cap+1e-12:raise RuntimeError('Approved API budget would be exceeded; no request sent')
        attempt={'attempt_id':len(self.data['attempts'])+1,'call_id':call_id,'status':'reserved_before_request',
            'reserved_usd':self.bound,'actual_cost_usd':None,'started_at':rt.now()}
        self.data['attempts'].append(attempt);rt.atomic(self.path,self.data)
        return attempt['attempt_id']

    def complete(self,attempt_id,usage_cost,*,status='completed_response',error=None):
        attempt=self.data['attempts'][attempt_id-1]
        if usage_cost is not None and (usage_cost<0 or usage_cost>attempt['reserved_usd']+1e-12):
            raise ValueError('Actual cost exceeded pre-request bound')
        attempt.update(status=status,actual_cost_usd=usage_cost,finished_at=rt.now(),error=error)
        rt.atomic(self.path,self.data)

    def reconcile(self,attempt_id,cost):
        # A durable response may survive a crash between response journaling and budget reconciliation.
        self.complete(attempt_id,cost,status='completed_response_reconciled_from_journal')
