import unittest
import gate


class CpuProofIdentityTests(unittest.TestCase):
    def setUp(self):
        self.validate=getattr(gate,'cpu_proof_identity',None)
        self.assertIsNotNone(self.validate,'CPU proof must bind config, verifier and all validator dependencies')
        self.proof={'cpu_validator_version':2,'adapter_config_sha256':'config',
            'verifier_sha256':'verifier','validator_manifest_sha256':'manifest'}

    def call(self):return self.validate(self.proof,'config','config','verifier','manifest')

    def test_current_complete_identity_accepted(self):self.assertTrue(self.call())

    def test_stale_configuration_rejected(self):
        self.proof['adapter_config_sha256']='old-config'
        self.assertFalse(self.call())

    def test_stale_validator_dependencies_rejected(self):
        self.proof['validator_manifest_sha256']='old-manifest'
        self.assertFalse(self.call())

    def test_stale_verifier_rejected(self):
        self.proof['verifier_sha256']='old-script'
        self.assertFalse(self.call())

    def test_wrong_version_or_completion_rejected(self):
        self.proof['cpu_validator_version']=1;self.assertFalse(self.call())
        self.proof['cpu_validator_version']=2
        self.assertFalse(self.validate(self.proof,'config','old-config','verifier','manifest'))


if __name__=='__main__':unittest.main()
