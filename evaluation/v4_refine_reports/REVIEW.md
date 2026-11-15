# V4 iteration review

The candidate retains public spellF1 0.5935 while improving macroF1 from 0.7261 to
0.7929. Overall dead-ball recall rises from 0.3996 to 0.6413, and visible-ball dead
recall rises from 0.1190 to 0.4831. This is a framewise improvement, not a new high
for the primary ranking metric. The strict release gate rejects promotion and V2
remains the default.

The structural regression is reduced but not solved. At IoU 0.5, V2 matches two
of six short (<5-second) true spells, V3 matches none, and V4 matches one. V4 loses
one previously matched short spell while matching ten total true spells, equal
to V2. Prediction count returns to 14 (V3: 9; GT: 17). Equal aggregate F1 does not
mean every original successful interval was preserved.

Two exploratory stages were run: 42 base configurations, then 21 announced
additions at intermediate switching penalties, 63 unique configurations total.
After a publication lifecycle fix, the same 63 were rerun without adding any
configurations; results were unchanged. Both studies are public-data development,
and held-out results retain historical label and design exposure.

Validation fixes include pooled official training scores instead of mean scores
on separately trimmed fragments, an actual frozen V2 candidate, and a held-out
mean-F1 promotion condition. A rejected or ordinary experiment now writes only
study-local candidate files, never an already-deployed config or default manifest.

27 tests passed, including exact V2 reproduction and publication lifecycle tests.
Canonical candidate inference and official CSV validation completed. Neither V2
code/config nor the official scorer was changed. Existing notebooks were untouched.

The next unresolved issue is distinguishing short genuine changes of control from
brief contacts and stoppages. Additional independently labeled sequences would be
more informative than repeatedly optimizing these same 17 public possessions.
