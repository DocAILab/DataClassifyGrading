import importlib
import math
import unittest


class Modules(unittest.TestCase):
    def mod(self, name):
        try: return importlib.import_module(name)
        except ModuleNotFoundError: self.fail(name + ' implementation is absent')

    def test_joint_outcome_requires_native_legal_prediction(self):
        m=self.mod('reward');ref={'category':'a','grading_level':'L2'}
        t={'terminated':True,'final_result':ref}
        self.assertEqual(m.outcome(t,ref,True),1)
        self.assertEqual(m.outcome(t,ref,False),0)
        self.assertEqual(m.outcome(dict(t,final_result={'category':'b','grading_level':'L2'}),ref,True),0)
        self.assertEqual(m.outcome(dict(t,final_result=None),ref,False),0)
        self.assertEqual(m.outcome(dict(t,terminated=False),ref,True),0)

    def test_wrong_long_chain_has_no_gated_dependency_bonus(self):
        m=self.mod('reward')
        a=m.combine(0,0,8,32);b=m.combine(0,0,16,32)
        self.assertAlmostEqual(a['A3'],.05);self.assertAlmostEqual(b['A3'],.1)
        self.assertEqual(a['A4'],0);self.assertEqual(b['A4'],0)
        self.assertAlmostEqual(m.combine(1,2,8,32)['A4'],1-.3*2/32+.2*8/32)

    def test_fixed_budget_and_invalid_counts_validated(self):
        m=self.mod('reward')
        for values in [(1,33,0,32),(1,-1,0,32),(1,0,33,32),(2,0,0,32)]:
            with self.assertRaises(ValueError):m.combine(*values)

    def test_first_missing_query_is_not_invalid(self):
        m=self.mod('reward');q=m.QueryLedger()
        self.assertFalse(q.before('field-description','v1'))
        q.after('field-description','v1','missing')
        self.assertTrue(q.before('field-description','v1'))
        self.assertFalse(q.before('field-description','v2'))

    def test_obtained_repeat_and_transient_retry(self):
        m=self.mod('reward');q=m.QueryLedger()
        q.after('q','v1','transient_error')
        self.assertFalse(q.before('q','v1'))
        q.after('q','v1','obtained')
        self.assertTrue(q.before('q','v1'))
        self.assertFalse(q.before('q2','v1'))

    def test_actual_versioned_dataflow_and_invalidation(self):
        m=self.mod('dataflow');g=m.VersionedFlow()
        g.add('s1',[],['metadata.x']);g.add('s2',['metadata.x'],['semantics.x'])
        g.add('unused',[],['noise']);g.add('final',['semantics.x'],[],terminal=True)
        self.assertEqual(g.ancestors('final'),{'s1','s2'})
        g.invalidate(['semantics.x']);g.add('final2',['semantics.x'],[],terminal=True)
        self.assertEqual(g.ancestors('final2'),set())

    def test_overwritten_data_does_not_keep_old_producer(self):
        m=self.mod('dataflow');g=m.VersionedFlow()
        g.add('old',[],['x']);g.add('new',[],['x']);g.add('final',['x'],[],terminal=True)
        self.assertEqual(g.ancestors('final'),{'new'})

    def test_failed_and_automatic_sink_not_rewarded(self):
        m=self.mod('dataflow');g=m.VersionedFlow()
        g.add('ok',[],['x']);g.add('bad',['x'],['y'],success=False)
        g.add('auto',['x'],[],terminal=True,automatic=True)
        self.assertEqual(g.rewarded('auto'),{'ok'})
        self.assertNotIn('bad',g.rewarded('auto'))

    def test_all_wrong_groups_masked_and_batch_std_finite(self):
        m=self.mod('grpo_math')
        x=m.advantages([[.1,.2],[1.,1.2]],[[0,0],[1,1]])
        self.assertEqual(x['active'],[[False,False],[True,True]])
        self.assertEqual(x['advantages'][0],[0.,0.])
        self.assertLess(x['advantages'][1][0],0);self.assertGreater(x['advantages'][1][1],0)
        self.assertTrue(all(math.isfinite(v) for rr in x['advantages'] for v in rr))

    def test_equal_rewards_have_zero_advantages(self):
        m=self.mod('grpo_math');x=m.advantages([[1,1,1,1]],[[1,1,1,1]])
        self.assertEqual(x['advantages'],[[0.,0.,0.,0.]])

    def test_loss_masks_external_tokens_and_uses_clip(self):
        m=self.mod('grpo_math')
        x=m.scalar_loss([0.,10.],[0.,0.],[0.,0.],[1.,99.],[1,0],clip=.2,beta=0)
        self.assertAlmostEqual(x,-1)
        self.assertAlmostEqual(m.scalar_loss([math.log(2)],[0],[0],[1],[1],clip=.2,beta=0),-1.2)


if __name__=='__main__':unittest.main()
