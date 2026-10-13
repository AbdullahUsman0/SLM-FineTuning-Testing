"""Deterministic grouped alternating GPU schedule, with hosted work after each block."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import runtime as rt
from registration import verify_final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','config','freeze'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();verify_final(args.freeze);config=rt.read(args.config)
    protocol=rt.read(rt.CHECKPOINT/'experiment-protocol.json')
    args.run.mkdir(parents=True,exist_ok=True)
    with rt.lock(args.run/'schedule.lock'):
        for block in protocol['schedule']:
            if (args.run/'stop.request').exists():raise SystemExit('Stop request observed between blocks')
            arm=block['arm'];group=block['group']
            env=dict(os.environ)
            if arm!='openai':env.pop('OPENAI_API_KEY',None)
            env.update(HF_HOME=config['HF_HOME'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
            with (args.run/'console.log').open('ab') as stdout,(args.run/'stderr.log').open('ab') as stderr:
                process=subprocess.run([config['python'],'-B','-u',str(rt.CHECKPOINT/'code/evaluate.py'),
                    '--run',str(args.run),'--freeze',str(args.freeze),'--config',str(args.config),
                    '--arm',arm,'--group',group],env=env,stdout=stdout,stderr=stderr)
            if process.returncode:raise SystemExit('Block stopped; verified completed calls remain resumable')
        # Finalization performs independent verification, not another inference pass.
        result=subprocess.run([config['python'],'-B','-u',str(rt.CHECKPOINT/'code/verify_results.py'),
            '--run',str(args.run),'--freeze',str(args.freeze),'--config',str(args.config)])
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
