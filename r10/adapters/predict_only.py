"""In-memory inference boundary: no extraction, GT scoring or config discovery."""
import numpy as np
from r10.tactical_tag_evidence import soft_team_support


def predict_only(alive, home, *, frame, period, fps, baseline_decoder,
                 tactical=None, tag_logit_weight=0.):
    """Call an explicitly supplied frozen decoder; no built-in selected model.

    Disabled tags give element-identical input heads to the same decoder.
    Enabled tags are experimental uncalibrated H/A evidence, never D evidence.
    No performance claim or production promotion follows from this interface.
    """
    q, h = np.array(alive, float, copy=True), np.array(home, float, copy=True)
    f, p = np.asarray(frame), np.asarray(period)
    if q.ndim != 1 or any(x.shape != q.shape for x in (h, f, p)):
        raise ValueError('unaligned heads')
    if (not np.isfinite(f).all() or not np.isfinite(p).all() or
            np.any(f != np.floor(f)) or np.any(p != np.floor(p)) or
            np.any(f < 0) or np.any(p < 0)):
        raise ValueError('invalid frame/period timeline')
    if (not np.isfinite(q).all() or not np.isfinite(h).all() or
            np.any((q < 0) | (q > 1) | (h < 0) | (h > 1)) or
            not np.isfinite([fps, tag_logit_weight]).all() or fps <= 0 or tag_logit_weight < 0):
        raise ValueError('invalid probability/time/weight')
    if tactical is not None and tag_logit_weight:
        support = soft_team_support(tactical, frame=f, period=p)['home_minus_away']
        changed = support != 0
        clipped = np.clip(h[changed], 1e-12, 1-1e-12)
        odds = np.log(clipped)-np.log1p(-clipped)+tag_logit_weight*support[changed]
        h[changed] = 1/(1+np.exp(-np.clip(odds, -700, 700)))
    labels = np.asarray(baseline_decoder(q, h, f.copy(), p.copy(), fps))
    if labels.shape != q.shape or not np.isin(labels, ['H', 'A', 'D', 'X']).all():
        raise ValueError('invalid decoder output')
    return labels.copy()
