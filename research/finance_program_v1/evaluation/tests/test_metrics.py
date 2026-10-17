import unittest
try:
    from metrics import score, paired
except ImportError:
    score=paired=None


class MetricTests(unittest.TestCase):
    def test_missing_nonterminal_wrong_stay_in_denominator(self):
        self.assertIsNotNone(score)
        ids=['a','b','c'];gold={i:{'category':'A','grading_level':'L1'} for i in ids}
        traces={'a':{'terminated':True,'final_result':gold['a']},'b':{'terminated':False,'final_result':gold['b']}}
        r=score(ids,traces,gold,['A','B'])
        self.assertEqual(r['denominator'],3)
        self.assertEqual(r['category_correct'],1)
        self.assertEqual(r['joint_correct'],1)
        self.assertEqual(r['missing'],1)
        self.assertAlmostEqual(r['category_accuracy'],1/3)

    def test_paired_retains_regressions(self):
        self.assertIsNotNone(paired)
        a={'x':{'category_correct':True},'y':{'category_correct':False}}
        b={'x':{'category_correct':False},'y':{'category_correct':True}}
        p=paired(['x','y'],a,b)['category_correct']
        self.assertEqual(p['sft_only'],['x']);self.assertEqual(p['base_only'],['y']);self.assertEqual(p['delta'],0)

    def test_only_retrieval_steps_define_top5_history(self):
        self.assertIsNotNone(score)
        def step(tool,correct):return {'call':{'tool':tool},'status':'success','data':{'standard_candidates':[{'category':x} for x in (['A','B','C','D','E'] if correct else ['B','C','D','E','F'])]}}
        trace={'terminated':False,'programs':[{'steps':[step('SELECT_CATEGORY_TOPK',True),step('UNRELATED',False)]}]}
        r=score(['x'],{'x':trace},{'x':{'category':'A','grading_level':'L1'}},['A','B','C','D','E','F'])
        self.assertEqual(r['top5_final_hit'],1)
