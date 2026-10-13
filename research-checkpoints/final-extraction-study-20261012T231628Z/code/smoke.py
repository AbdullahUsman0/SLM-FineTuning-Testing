"""GPU readiness on separate fixtures only; never read candidate/final examples."""
import argparse
import gc
from pathlib import Path
import runtime as rt
from providers import Local
from shared_policy import SharedPolicy

FIXTURES=[
    ('unknown-before-goal','I have no idea which measurements are available.',None),
    ('basic','Please forecast midday dew-point temperature. I can upload iris_smoke.csv; its value field is dew_reading and its timestamp field is instant_utc.',None),
    ('literal','Use the literal column name "2026" as the target identifier, not the number 2026.',None),
    ('nested','My scope_filters requirement is [{"site":"trial","sensors":["A","B"]}]. Keep the list order.',None),
    ('whitespace','Please\n  forecast pond surface temperature. Give forecasts\n  in CSV format.',None),
    ('correction','Replace the 5-hour horizon with 12 hours; leave every other recorded requirement alone.',
      {'intent':'create_forecast','intent_confidence':1.0,'correction_detected':False,'unsupported_claims':[],
       'updates':[{'slot_id':'forecast_horizon','candidate_value':'{"periods":5,"unit":"hour"}',
                   'status':'provided','confidence':1.0,'evidence_text':'5 hours'}]})]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=rt.read(args.config)
    rt.verify_sources();rt.verify_models(config)
    schema=rt.load_schema();policy=SharedPolicy()
    out=Path(config['private'])/('smoke-'+rt.now().replace(':','').replace('+','_'))
    out.mkdir(exist_ok=False)
    rt.atomic(out/'fixtures.json',{'scope':'separate unscored smoke fixtures, not the final cohort',
        'fixtures':FIXTURES,'fixtures_sha256':rt.digest(FIXTURES),'paid_API_calls':0},exclusive=True)
    results={}
    for arm in ('v6','v7'):
        print('Loading '+arm+' for CUDA readiness.',flush=True)
        provider=Local(arm,config)
        warmup=provider.generate('Forecast ground-level ozone for 5 hours.',rt.create_initial_state(schema),schema)
        rt.atomic(out/(arm+'-warmup.json'),warmup,exclusive=True)
        rt.atomic(out/(arm+'-runtime.json'),provider.inventory,exclusive=True)
        records=[]
        for fixture_id,message,prior in FIXTURES:
            state=rt.create_initial_state(schema)
            if prior:
                parsed=policy.parse(rt.dumps(prior),'5 hours')
                state,accepted,error=policy.advance(state,parsed,schema,1,'5 hours');assert accepted
                rt.append_user(state,'5 hours')
            raw=provider.generate(message,state,schema)
            admission=policy.parse(raw['raw_output'],message,response_status=raw['response_status'],
                refused=raw['refused'],hit_generation_limit=raw['hit_generation_limit'])
            record={'fixture_id':fixture_id,'message':message,'raw':raw,'admission':admission.audit()}
            records.append(record);rt.atomic(out/(arm+'-'+fixture_id+'.json'),record,exclusive=True)
            print(arm+' '+fixture_id+': '+str(admission.error_code or 'admitted'),flush=True)
        results[arm]={'CUDA_verified':True,'active_adapters':provider.inventory['active_adapters'],
            'runtime':provider.inventory,'fixtures':len(records),'warmup_calls':1,
            'semantic_or_contract_failures_are_recorded_not_retried':True,
            'generation_limit_hits':sum(r['raw']['hit_generation_limit'] for r in records)}
        del provider;gc.collect()
        import torch
        torch.cuda.empty_cache()
    rt.verify_models(config)
    receipt={'status':'GPU_readiness_verified','results':results,'scope':'separate fixtures, not final scoring',
        'paid_API_calls':0,'cohort_examples_consumed':0,'training_started':False,'output_directory':str(out),
        'completed_at':rt.now(),'fixtures_sha256':rt.digest(FIXTURES)}
    rt.atomic(out/'completion.json',receipt,exclusive=True)
    rt.atomic(rt.CHECKPOINT/'GPU-SMOKE.json',receipt,exclusive=True)
    print(rt.dumps(receipt),flush=True)


if __name__=='__main__':main()
