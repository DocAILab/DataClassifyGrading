import unittest
try:
    from replay import Recorded, validate_record
except ImportError:
    Recorded = validate_record = None


class RecordTests(unittest.TestCase):
    def test_missing_or_changed_request_not_silently_consumed(self):
        self.assertIsNotNone(Recorded)
        replay = Recorded([])
        with self.assertRaises(AssertionError):replay([{'role':'system','content':'x'}])
        with self.assertRaises(AssertionError):replay.finish()

    def test_unconsumed_call_fails(self):
        self.assertIsNotNone(Recorded)
        r=Recorded([{'request':{},'result':{}}])
        with self.assertRaises(AssertionError):r.finish()

    def test_route_must_follow_trusted_system(self):
        self.assertIsNotNone(validate_record)
        messages=[{'role':'system','content':'skill'}]
        q={'messages':messages,'generation':{'do_sample':False,'max_new_tokens':512,'max_time':90.,'eos_token_id':[248044,248046],'pad_token_id':248044}}
        x={'messages':messages,'generation':q['generation'],'raw_text':'{}','error':None,'input_tokens':2,'output_tokens':1,'truncation_at_budget':False,'input_devices':['cuda:0'],'parameter_devices':['cuda:0'],'parameter_dtypes':['torch.float16']}
        route={'arm':'sft','enabled':True,'layer_count':40,'entered_disabled':False,'restored_disabled':True}
        with self.assertRaises(AssertionError):validate_record(q,x,route,'sft','planner',None)


if __name__=='__main__':unittest.main()
