import unittest
try:
    from training_contract import expected_steps,validate_trial,validate_gpu_snapshot
except ImportError:
    expected_steps=validate_trial=validate_gpu_snapshot=None


class TrainingContractTests(unittest.TestCase):
    def test_last_eight_examples_retained(self):
        self.assertIsNotNone(expected_steps,'missing step gate')
        self.assertEqual(expected_steps(200,16),13)

    def test_trial_requires_actual_longest_source_and_data(self):
        self.assertIsNotNone(validate_trial,'missing trial gate')
        c={'data_sha256':'a','max_length':42,'longest_source_id':'x','gpu_uuids':{'0':'u0','1':'u1','2':'u2'}}
        p={'data_sha256':'a','tokens':42,'source_id':'x','gpu_uuids':c['gpu_uuids'],'base_and_vision_frozen':True,'lora_updated':True,'gradients_finite':True,'loss_finite':True,'target_modules':40}
        validate_trial(p,c)
        p['source_id']='wrong'
        with self.assertRaises(ValueError):validate_trial(p,c)

    def test_trial_rejects_nonfinite_or_no_update(self):
        self.assertIsNotNone(validate_trial,'missing trial gate')
        c={'data_sha256':'a','max_length':42,'longest_source_id':'x','gpu_uuids':{'0':'u0'}}
        p={'data_sha256':'a','tokens':42,'source_id':'x','gpu_uuids':c['gpu_uuids'],'base_and_vision_frozen':True,'lora_updated':False,'gradients_finite':True,'loss_finite':True,'target_modules':40}
        with self.assertRaises(ValueError):validate_trial(p,c)

    def test_busy_gpu_rejected_even_if_our_lock_is_free(self):
        self.assertIsNotNone(validate_gpu_snapshot,'missing resource gate')
        with self.assertRaises(ValueError):validate_gpu_snapshot({'0':'u0','1':'u1','2':'u2'}, {'0':'u0','1':'u1','2':'u2'}, {'0':32000,'1':32000,'2':32000},['u2'])

    def test_wrong_uuid_rejected(self):
        self.assertIsNotNone(validate_gpu_snapshot,'missing resource gate')
        with self.assertRaises(ValueError):validate_gpu_snapshot({'0':'u0','1':'u1','2':'u2'}, {'0':'u0','1':'u1','2':'wrong'}, {'0':32000,'1':32000,'2':32000},[])

    def test_idle_exact_three_gpu_gate(self):
        self.assertIsNotNone(validate_gpu_snapshot,'missing resource gate')
        validate_gpu_snapshot({'0':'u0','1':'u1','2':'u2'}, {'0':'u0','1':'u1','2':'u2'}, {'0':32000,'1':32000,'2':32000},[])


if __name__=='__main__':unittest.main()
