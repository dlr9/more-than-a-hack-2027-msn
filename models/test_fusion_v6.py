import unittest
from unittest.mock import patch
import numpy as np
import fusion_v6 as v6


class V6Tests(unittest.TestCase):
    def test_requires_events(self):
        with self.assertRaises(ValueError):v6.predict({})

    def test_applies_scheme_two_only_and_preserves_base(self):
        base=np.array(list('XHADX'))
        held=np.array([False,True,False,False,False])
        with patch.object(v6.v5,'predict',return_value=base) as pred, patch.object(v6,'masks',return_value=(held,[])) as masks:
            result=v6.predict({'restart_events':[]})
            self.assertEqual(''.join(result),'XDADX')
            self.assertEqual(''.join(base),'XHADX')
            self.assertEqual(pred.call_args.args[1]['restart_experiment'],'none')
            self.assertEqual(masks.call_args.args[-1],'tag_head')

if __name__=='__main__':unittest.main()
