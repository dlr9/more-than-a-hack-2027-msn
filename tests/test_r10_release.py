import unittest
from scripts.audit_public_release import allowed,inspect_blob
from scripts.check_public_tree import violations

class ReleaseBoundaryTests(unittest.TestCase):
    def test_only_source_and_exact_ledger_allowed(self):
        for p in ['r10/acquisition.py','r10/adapters/tracking.py','r10/adapters/B_to_R9_contract.md','docs/R10_V3_DECISIONS.jsonl']:
            self.assertTrue(allowed(p))
        for p in ['r10/assets/input.npz','r10/adapters/data/target.py','docs/R10_V3_other.jsonl','local_artifacts/model.joblib','r10/../private.md']:
            self.assertFalse(allowed(p))
        self.assertEqual(violations(['docs/R10_V3_DECISIONS.jsonl']),[])
        self.assertTrue(violations(['docs/R10_V3_other.jsonl']))
    def test_serialized_prediction_rejected_even_as_source(self):
        with self.assertRaises(ValueError):inspect_blob('r10/adapters/test.py',b'frame,label\n0,H\n')
if __name__=='__main__':unittest.main()
