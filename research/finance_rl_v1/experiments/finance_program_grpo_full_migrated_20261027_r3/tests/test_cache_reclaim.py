import unittest,resource_runtime as m
class CacheReclaim(unittest.TestCase):
 def test_only_unused_cache_is_released(self):
  self.assertTrue(hasattr(m,'release_cache'),'cache release boundary missing')
  calls=[]
  class C:
   def device(self,i):
    class X:
     def __enter__(self):calls.append(('device',i))
     def __exit__(self,*a):pass
    return X()
   def empty_cache(self):calls.append(('empty',))
  class T:cuda=C()
  self.assertTrue(m.release_cache(T(),[{'reserved_bytes':1024**3,'allocated_bytes':0,'free_bytes':10*1024**3}]*3))
  self.assertEqual(sum(x[0]=='empty' for x in calls),3)
 def test_no_cache_clear_when_all_memory_in_use_and_sufficient_free(self):
  self.assertTrue(hasattr(m,'release_cache'),'cache release boundary missing')
  self.assertFalse(m.release_cache(None,[{'reserved_bytes':1024**3,'allocated_bytes':1024**3,'free_bytes':10*1024**3}]*3))
