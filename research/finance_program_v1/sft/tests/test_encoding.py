import unittest

try:
    from token_audit import encode_training_row
except ImportError:
    encode_training_row=None


class Tokenizer:
    eos_token="!"
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return "P:"
    def __call__(self,text,**kwargs):
        assert kwargs==dict(add_special_tokens=False)
        return {"input_ids":[ord(c) for c in text]}


class EncodingTests(unittest.TestCase):
    def test_only_response_and_eos_supervised(self):
        self.assertIsNotNone(encode_training_row,"missing encoding feature")
        r=encode_training_row({"source_id":"a","messages":[],"response":"{}"},Tokenizer(),5)
        self.assertEqual(r["labels"],[-100,-100,ord('{'),ord('}'),ord('!')])
        self.assertEqual(len(r["input_ids"]),5)

    def test_overlength_rejected_not_truncated(self):
        self.assertIsNotNone(encode_training_row,"missing encoding feature")
        with self.assertRaises(ValueError):encode_training_row({"source_id":"a","messages":[],"response":"{}"},Tokenizer(),4)

    def test_missing_eos_rejected(self):
        self.assertIsNotNone(encode_training_row,"missing encoding feature")
        tok=Tokenizer();tok.eos_token=None
        with self.assertRaises(ValueError):encode_training_row({"source_id":"a","messages":[],"response":"{}"},tok,5)

    def test_boundary_change_rejected(self):
        self.assertIsNotNone(encode_training_row,"missing encoding feature")
        class Bad(Tokenizer):
            def __call__(self,text,**kwargs):return {"input_ids":[1] if text=="P:" else [2,3,4]}
        with self.assertRaises(ValueError):encode_training_row({"source_id":"a","messages":[],"response":"{}"},Bad(),5)


if __name__=='__main__':unittest.main()
