import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from io_utils import read
from backend import RoutedBackend


class RoutingStateTests(unittest.TestCase):
    def test_generation_error_always_restores_disabled_layers(self):
        class Model:
            def __init__(self):
                self.layers=[SimpleNamespace(lora_A={},disable_adapters=True,active_adapters=['default']) for _ in range(40)]
                self.base_model=self
            def named_modules(self):return [(str(i),x) for i,x in enumerate(self.layers)]
            def disable_adapter_layers(self):
                for x in self.layers:x.disable_adapters=True
            def enable_adapter_layers(self):
                for x in self.layers:x.disable_adapters=False
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):raise ValueError('controlled encoding failure')
        for arm,role in (('base','planner'),('sft','planner'),('sft','skill')):
            with tempfile.TemporaryDirectory() as d:
                b=RoutedBackend.__new__(RoutedBackend);b.model=Model();b.system='planner';b.arm=arm
                b.call_dir=Path(d);b.calls=[];b.failures=[];b.options={};b.tokenizer=Tokenizer()
                b.parameter_devices=['cuda:0'];b.parameter_dtypes=['torch.float16'];b.anchor={'adapter_sha256':'a'}
                with self.assertRaises(ValueError):b.generate([{'role':'system','content':role}])
                route=read(Path(d)/'call-0001/route.json')
                self.assertEqual(route['enabled'],arm=='sft' and role=='planner')
                self.assertTrue(route['restored_disabled'])
                self.assertTrue(all(x.disable_adapters for x in b.model.layers))
