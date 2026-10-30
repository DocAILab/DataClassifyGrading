"""Full-context frozen local generation with measured Planner-only LoRA routing."""
from copy import deepcopy
from pathlib import Path
import time
from io_utils import ROOT, read, put, sha
from gate import enabled_for,eos_options,validate_adapter_files
from local_smoke import Backend,encode_messages,generation_metadata


def layer_state(model,disabled):
    layers=[m for _,m in model.named_modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')]
    if len(layers)!=40 or any(bool(m.disable_adapters)!=disabled or list(m.active_adapters)!=['default'] for m in layers):
        raise RuntimeError('actual LoRA layer state differs from requested route')
    return {'layer_count':len(layers),'disabled':disabled}


class RoutedBackend(Backend):
    def __init__(self,model,adapter,anchor,cohort):
        if read(Path(model)/'model-manifest.json')!=read(ROOT/'model-identity.json'):
            raise ValueError('base model manifest differs from frozen original identity')
        validate_adapter_files(adapter,anchor)
        super().__init__(model,ROOT/'runs'/(cohort+'-backend-init'))
        from peft import PeftModel
        validate_adapter_files(adapter,anchor)
        self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=False,autocast_adapter_dtype=False)
        validate_adapter_files(adapter,anchor)
        self.model.to(dtype=self.torch.float16);self.model.requires_grad_(False);self.model.eval()
        self.parameter_devices=sorted({str(p.device) for p in self.model.parameters()})
        self.parameter_dtypes=sorted({str(p.dtype) for p in self.model.parameters() if p.is_floating_point()})
        if self.parameter_devices!=['cuda:0'] or self.parameter_dtypes!=['torch.float16']:
            raise RuntimeError('entire model and adapter must be FP16 cuda:0')
        self.options['eos_token_id']=eos_options(self.model.generation_config.eos_token_id,self.tokenizer.eos_token_id)
        if self.options['eos_token_id']!=[248044,248046]:raise ValueError('model/tokenizer EOS identity differs')
        self.system=read(ROOT/'data/protocol.json')['planner_system'];self.anchor=anchor
        self.model.base_model.disable_adapter_layers();layer_state(self.model,True)
        put(ROOT/'runs'/(cohort+'-backend-loaded.json'),{'adapter':str(adapter),'adapter_sha256':anchor['adapter_sha256'],
            'base_revision':self.identity['revision'],'parameter_devices':self.parameter_devices,
            'parameter_dtypes':self.parameter_dtypes,'generation':self.options,'targets':40})

    def start_source(self,folder,arm):
        self.arm=arm;self.call_dir=Path(folder)/'calls';self.call_dir.mkdir(parents=True,exist_ok=False)
        self.calls=[];self.failures=[]

    def generate(self,messages):
        enabled=enabled_for(self.arm,messages,self.system)
        call=self.call_dir/('call-%04d'%(len(self.calls)+1));call.mkdir();self.calls.append(str(call))
        self.last_generation_metadata={};started=time.perf_counter()
        q={'messages':deepcopy(messages),'generation':self.options};put(call/'request.json',q)
        x={'messages':deepcopy(messages),'raw_text':None,'input_tokens':None,'output_tokens':None,
            'generated_ids':[],'input_devices':[],'parameter_devices':self.parameter_devices,
            'parameter_dtypes':self.parameter_dtypes,'generation':self.options,'truncation_at_budget':False,'error':None}
        route={'arm':self.arm,'enabled':enabled,'layer_count':40,'entered_disabled':None,'restored_disabled':None,
            'adapter_sha256':self.anchor['adapter_sha256']}
        try:
            if enabled:self.model.base_model.enable_adapter_layers()
            else:self.model.base_model.disable_adapter_layers()
            try:
                state=layer_state(self.model,not enabled);route['entered_disabled']=state['disabled']
                encoded=encode_messages(self.tokenizer,messages)
                x['input_tokens']=encoded['input_ids'].shape[1]
                encoded={k:v.to('cuda:0') for k,v in encoded.items()};x['input_devices']=['cuda:0']
                self.torch.cuda.synchronize(0)
                with self.torch.inference_mode():output=self.model.generate(**encoded,**self.options)
                self.torch.cuda.synchronize(0)
                ids=output[0,x['input_tokens']:].tolist();x['generated_ids']=ids;x['output_tokens']=len(ids)
                x['truncation_at_budget']=len(ids)>=512
                x['raw_text']=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                return x['raw_text']
            finally:
                self.model.base_model.disable_adapter_layers()
        except Exception as exc:
            x['error']=type(exc).__name__+': '+str(exc);self.failures.append(x['error']);raise
        finally:
            route['restored_disabled']=layer_state(self.model,True)['disabled']
            x['seconds']=time.perf_counter()-started
            self.last_generation_metadata=generation_metadata(x)
            put(call/'result.json',x)
            route.update(request_sha256=sha(call/'request.json'),result_sha256=sha(call/'result.json'))
            put(call/'route.json',route)
