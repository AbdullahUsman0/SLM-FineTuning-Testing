"""Provider transports share instructions and use the same downstream admission."""
import os
from pathlib import Path
from time import perf_counter
import runtime as rt
from shared_policy import SharedPolicy

OPENAI_MODEL='gpt-5-mini-2025-08-07'
OPENAI_OUTPUT_CAP=2048
API_INPUT_BYTES_CAP=32768
API_TOKEN_OVERHEAD_BOUND=1024
PRICES={'input_per_million':0.25,'cached_input_per_million':0.025,'output_per_million':2.0}


def hosted_request(message,state,schema,policy):
    return {'model':OPENAI_MODEL,'store':False,'reasoning':{'effort':'minimal'},
        'max_output_tokens':OPENAI_OUTPUT_CAP,
        'input':[{'role':'system','content':rt.instructions()},
                 {'role':'user','content':rt.context(message,state,schema)}],
        'text':{'format':{'type':'json_schema','name':'forecast_requirements','strict':True,'schema':policy.schema}}}


def usage_cost(usage):
    if not usage:
        return None
    cached=(usage.get('input_tokens_details') or {}).get('cached_tokens',0)
    return ((usage['input_tokens']-cached)*PRICES['input_per_million']+
             cached*PRICES['cached_input_per_million']+
             usage['output_tokens']*PRICES['output_per_million'])/1_000_000


def worst_attempt_cost():
    # UTF-8 bytes conservatively bound BPE input tokens, plus declared protocol overhead allowance.
    return ((API_INPUT_BYTES_CAP+API_TOKEN_OVERHEAD_BOUND)*PRICES['input_per_million']+
             OPENAI_OUTPUT_CAP*PRICES['output_per_million'])/1_000_000


def decode_hosted(response):
    texts=[];refused=False
    for output in response.get('output',[]):
        if output.get('type')!='message':continue
        for content in output.get('content',[]):
            if content.get('type')=='output_text':texts.append(content.get('text',''))
            if content.get('type')=='refusal':refused=True
    usage=response.get('usage')
    return {'raw_output':''.join(texts),'response_status':response.get('status','missing'),
        'refused':refused,'hit_generation_limit':(response.get('incomplete_details') or {}).get('reason')=='max_output_tokens',
        'prompt_tokens':None if not usage else usage.get('input_tokens'),
        'completion_tokens':None if not usage else usage.get('output_tokens'),
        'reasoning_tokens':0 if not usage else (usage.get('output_tokens_details') or {}).get('reasoning_tokens',0),
        'usage':usage,'cost_usd':usage_cost(usage),
        'response':{key:response.get(key) for key in ('id','model','status','output','usage','incomplete_details')}}


class Local:
    def __init__(self,arm,config):
        rt.verify_sources('local');rt.verify_models(config,check_base=True)
        os.environ.update(HF_HOME=config['HF_HOME'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        from local_slm_lab.peft_provider import PeftInferenceConfig
        from local_slm_lab.v5_provider import V5TransformersProvider
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
        torch.set_num_threads(4)
        self.provider=V5TransformersProvider(PeftInferenceConfig(base_model='Qwen/Qwen3.5-2B',
            revision=rt.BASE_REVISION,variant='custom',adapter_path=Path(config['adapters'][arm]),
            device='cuda',dtype='bfloat16',max_new_tokens=1024),rt.load_schema())
        self.torch=torch
        model=self.provider._model
        if any(p.device.type!='cuda' for p in model.parameters()):raise RuntimeError('CPU offload is prohibited')
        if any(p.dtype!=torch.bfloat16 for name,p in model.named_parameters() if p.is_floating_point() and 'lora_' not in name):
            raise RuntimeError('Loaded base-model precision differs from frozen BF16')
        if any(p.dtype not in {torch.bfloat16,torch.float32} for name,p in model.named_parameters() if 'lora_' in name):
            raise RuntimeError('Unexpected PEFT adapter precision')
        active=model.active_adapters
        if not active or getattr(model,'_disable_adapters',False):raise RuntimeError('PEFT adapter is inactive')
        self.inventory={'model_class':type(model).__name__,'processor_class':type(self.provider._processor).__name__,
            'tokenizer_class':type(getattr(self.provider._processor,'tokenizer',self.provider._processor)).__name__,
            'chat_template_sha256':self.provider.chat_template_sha256,
            'active_adapters':active,'peft_adapter_configs':{k:{'class':type(v).__name__,'r':v.r,
                'lora_alpha':v.lora_alpha,'target_modules':sorted(v.target_modules)} for k,v in model.peft_config.items()},
            'parameter_devices':sorted({str(p.device) for p in model.parameters()}),
            'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
            'base_commit_hash':getattr(model.config,'_commit_hash',None),'revision_requested':rt.BASE_REVISION,
            'gpu':torch.cuda.get_device_name(0),'precision':'BF16 base; PEFT default FP32 LoRA parameters retained, same verified loading path for both arms',
            'cpu_offload':False}
        if self.inventory['base_commit_hash']!=rt.BASE_REVISION:raise RuntimeError('Loaded base revision was not verified')

    def generate(self,message,state,schema):
        self.torch.cuda.synchronize();started=perf_counter()
        raw,prompt_tokens,completion_tokens,limit=self.provider._generate_raw(rt.instructions(),rt.context(message,state,schema))
        self.torch.cuda.synchronize()
        return {'raw_output':raw,'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,
            'hit_generation_limit':limit,'response_status':'completed','refused':False,
            'generation_elapsed_seconds':perf_counter()-started,'cost_usd':0,'reasoning_tokens':0}


class Hosted:
    def __init__(self):
        rt.require_study_execution()
        rt.verify_sources('openai')
        from openai import OpenAI
        if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Configure OPENAI_API_KEY securely; never put it in manifests')
        self.client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url='https://api.openai.com/v1',max_retries=0,timeout=90.0)
        available=self.client.models.retrieve(OPENAI_MODEL)
        if available.id!=OPENAI_MODEL:raise RuntimeError('Pinned hosted snapshot availability was not confirmed')
        self.inventory={'model':OPENAI_MODEL,'API':'Responses','structured_outputs':True,'reasoning_effort':'minimal',
            'output_token_cap_including_reasoning':OPENAI_OUTPUT_CAP,'SDK_automatic_retries':0,'concurrency':1,
            'snapshot_access_confirmed_before_paid_requests':True}

    def generate(self,message,state,schema,policy):
        rt.require_study_execution()
        request=hosted_request(message,state,schema,policy)
        if len(rt.dumps(request).encode('utf-8'))>API_INPUT_BYTES_CAP:raise RuntimeError('Frozen API input bound exceeded; no request sent and no truncation')
        started=perf_counter()
        result=self.client.responses.create(**request)
        record=decode_hosted(result.model_dump(mode='json'))
        record['generation_elapsed_seconds']=perf_counter()-started
        return record
