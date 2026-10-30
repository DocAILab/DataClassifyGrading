import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'finance_program_skill_recovery20_20261021_r1'
path=ROOT/'interface.py' if (ROOT/'interface.py').exists() else OLD/'interface.py'
s=importlib.util.spec_from_file_location('under_test',path)
mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
CORPUS=OLD/'reviews/final-01/extract/runs'

class InterfaceTests(unittest.TestCase):
    def test_actual_overflow_is_lossless_and_smaller(self):
        record=json.loads((CORPUS/'native-19-unified/call-005/returned.json').read_bytes())
        actual=json.loads((CORPUS/'native-19-unified/call-005/model-returned.json').read_bytes())
        rewritten=mod.rewrite(record['messages'])
        packet=json.loads(rewritten[-1]['content'])
        self.assertTrue('lossless_message_values' in packet,'recovery payload still repeatedly expands identical observations')
        self.assertLess(len(rewritten[-1]['content']),len(actual['messages'][-1]['content'])*0.8)
        self.assertEqual(mod.expand_packet(packet),json.loads(actual['messages'][-1]['content']))
        self.assertEqual(record['messages'],json.loads((CORPUS/'native-19-unified/call-005/returned.json').read_bytes())['messages'])

    def test_actual_selector_string_list_prompt(self):
        rec=json.loads((CORPUS/'native-09-unified/call-006/returned.json').read_bytes())
        rewritten=mod.rewrite(rec['messages'])
        self.assertIn('selected_context_fields MUST be a nonempty JSON array of distinct field-name strings',rewritten[0]['content'])
        self.assertIn('not objects',rewritten[0]['content'])
        self.assertEqual(rewritten[-1],rec['messages'][-1])

    def test_reference_codec_preserves_key_order_and_rejects_bad_refs(self):
        self.assertTrue(hasattr(mod,'pack'),'lossless codec not implemented')
        value={'z':['same'*100]*3,'a':{'support':False,'empty':None,'steps':list(range(20))},'b':['same'*100]*3}
        encoded=mod.pack(value)
        self.assertEqual(json.dumps(mod.expand_packet(encoded)),json.dumps(value))
        for bad in [{'lossless_message_values':{'r0':{'$message_ref':'r0'}},'payload':{'$message_ref':'r0'}},
                    {'lossless_message_values':{},'payload':{'$message_ref':'missing'}}]:
            with self.assertRaises(ValueError):mod.expand_packet(bad)

    def test_nested_pretty_json_strings_are_lossless_not_opaque_duplicates(self):
        obj={'evidence':{'description':'真实字段描述'*50,'fields':list(range(10))},'status':'failed'}
        text=json.dumps(obj,ensure_ascii=False,indent=2)
        value={'original':obj,'raw_message':text,'another':text}
        packet=mod.pack(value)
        self.assertTrue('$message_json_string' in json.dumps(packet),'serialized original messages remain opaque duplicate strings')
        self.assertEqual(json.dumps(mod.expand_packet(packet),ensure_ascii=False),json.dumps(value,ensure_ascii=False))

if __name__=='__main__':unittest.main()
