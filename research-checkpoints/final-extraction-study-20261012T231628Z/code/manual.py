"""Interactive separate-fixture testing through the exact shared admission path."""
import argparse
from pathlib import Path
import runtime as rt
from shared_policy import SharedPolicy
from providers import Local


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--arm',choices=['v6','v7'],default='v7');args=parser.parse_args()
    provider=Local(args.arm,rt.read(args.config));schema=rt.load_schema();policy=SharedPolicy();state=rt.create_initial_state(schema)
    print('Shared-policy manual test on your own messages. Do not use protected final-cohort examples. Type /quit to exit.')
    while True:
        message=input('You: ')
        if message.strip()=='/quit':break
        raw=provider.generate(message,state,schema)
        admission=policy.parse(raw['raw_output'],message,response_status=raw['response_status'],
            refused=raw['refused'],hit_generation_limit=raw['hit_generation_limit'])
        print(raw['raw_output']);print('Admission: '+str(admission.error_code or 'accepted'))
        state,accepted,error=policy.advance(state,admission,schema,len(state.turns)+1,message)
        rt.append_user(state,message)
        print(rt.dumps({'application_accepted':accepted,'application_error':error,'state':rt.semantic_state(state)}))


if __name__=='__main__':main()
