import ast
from pathlib import Path
import unittest
import numpy as np
from solution.multimodal_evidence import EvidencePacket,EditProposal,apply_edits,residual_candidates

class IsolationTests(unittest.TestCase):
 def test_runtime_never_imports_oracle(self):
  for folder in ['solution','multiview']:
   for path in Path(folder).glob('*.py'):
    tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
     names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
     self.assertFalse(any(n.startswith('research_only') for n in names),str(path))
 def test_malformed_or_uncalibrated_edits_never_change_baseline(self):
  a=np.array(list('H'*100));period=np.ones(100,int)
  e=EvidencePacket(40,1,'synthetic','observed',.9,.1,'fixture',1.,'fixture')
  for label in ['', 'HA', 'X']:
   with self.assertRaises(ValueError):residual_candidates(e,label=label)
   q=EditProposal(38,43,label,e)
   np.testing.assert_array_equal(apply_edits(a,period,[q],new_feature=True,trust=1,model_off=False,calibrated_gate=lambda x:True,max_changed_fraction=.1),a)
  q=EditProposal(38,43,'A',e)
  self.assertIs(apply_edits(a,period,[q],new_feature=True,trust=1,model_off=False),a)
if __name__=='__main__':unittest.main()
