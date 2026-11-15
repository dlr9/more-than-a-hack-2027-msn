# V3 review and release decision

V3 is retained as an experimental model. The default remains V2 because the
primary ranking metric regressed from 0.5935 to 0.4923, despite macroF1 improving
from 0.7261 to 0.8192. No manual frame corrections or extra search rounds were used.

The targeted failure improved: recall on the 975 public dead-ball frames with a
visible ball increased from 11.9% to 57.5%. Overall dead-ball recall increased
from 40.0% to 70.3%. However, predicted public possession spells fell from 14 to
9 against 17 true spells. This is evidence of excessive merging or missed spells,
not proof that dead-ball recall alone solves the ranking objective.

The selected configuration uses fixed transition penalties (dynamic=0) and no
trajectory interpolation (gap=0). Those capabilities were implemented and tested
but were not selected. Their removal therefore yields identical ablation scores.

Validation completed:

- 17 V2/V3 tests, including exhaustive dynamic Viterbi path comparison.
- Canonical pipeline execution with V3 and official public evaluation.
- Official submission validation: 199319 contiguous frames, legal labels.
- Reused search fingerprint check and explicit promotion request: gate rejects V3.
- V2 model implementation and config remain unchanged.

Next scientific question: can visible-ball stoppage evidence improve interval
boundaries without losing short possession spells? The current data do not support
claiming improved full-match performance. Public scores, held-out blocks and
bootstrap intervals all remain affected by historical development exposure.
