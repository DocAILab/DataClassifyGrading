import unittest
import resource_profile as profile

UUIDS={0:'GPU-4e00b91b-5123-d78d-35f3-e72212f3f588',2:'GPU-4c58bc24-4fe1-36cb-55ce-e43f346d954c',3:'GPU-fa094e6f-f139-fc8f-dfc4-013092b3a716'}
def rows():
    return [dict(index=i,uuid=UUIDS.get(i,'other-'+str(i)),bf16_compatible=True,free_bytes=31*2**30,other_compute_pids=[]) for i in range(8)]
class PinTests(unittest.TestCase):
    def test_exact_user_cards_and_uuid_fail_closed(self):
        data=rows()
        self.assertEqual([c['gpu_indices'] for c in profile.candidates(data)],[[0,2,3]])
        anchor=profile.candidates(data)[0]
        data[2]['free_bytes']=29*2**30
        self.assertEqual(profile.candidates(data),[])
        with self.assertRaises(ValueError):profile.recheck(data,anchor)
        data=rows();data[3]['uuid']='replacement'
        self.assertEqual(profile.candidates(data),[])
if __name__=='__main__':unittest.main()
