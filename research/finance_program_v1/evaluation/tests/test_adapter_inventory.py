"""Actual PEFT0.21 serialized suffix reduction must expand to the original 40 layers."""
import copy
import unittest
from io_utils import ROOT,read
import gate


class AdapterInventoryTests(unittest.TestCase):
    def setUp(self):
        self.validate=getattr(gate,'validate_lora_inventory',None)
        self.assertIsNotNone(self.validate,'missing semantic inventory validator; serialized suffix count is not layer count')
        self.config=read(ROOT/'fixtures/adapter-config-diagnosis.json')['config']
        self.arch=read(ROOT/'fixtures/adapter-architecture.json')
        self.inventory=read(ROOT/'fixtures/adapter-inventory.json')['rows']

    def call(self):
        return self.validate(self.config,self.inventory,self.arch['formal_targets'],self.arch['all_linear_module_shapes'])

    def test_real_three_suffixes_resolve_to_all_forty_layers(self):
        self.assertEqual(len(self.config['target_modules']),3)
        result=self.call()
        self.assertEqual(result['resolved_targets'],40)
        self.assertEqual(result['tensor_count'],80)

    def test_missing_pair_rejected(self):
        self.inventory.pop()
        with self.assertRaises(ValueError):self.call()

    def test_wrong_shape_rejected(self):
        self.inventory[0]['shape']=[7,4096]
        with self.assertRaises(ValueError):self.call()

    def test_suffix_matching_extra_vision_module_rejected(self):
        self.arch['all_linear_module_shapes']['model.visual.q_proj']=[16,16]
        with self.assertRaises(ValueError):self.call()

    def test_changed_rank_or_dropout_rejected(self):
        self.config['r']=16
        with self.assertRaises(ValueError):self.call()
        self.config['r']=8;self.config['lora_dropout']=.2
        with self.assertRaises(ValueError):self.call()

    def test_extra_weight_or_duplicate_rejected(self):
        self.inventory.append(copy.deepcopy(self.inventory[0]))
        with self.assertRaises(ValueError):self.call()


if __name__=='__main__':unittest.main()
