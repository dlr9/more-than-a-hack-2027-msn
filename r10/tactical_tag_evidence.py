"""Soft tactical interval evidence; no files, target data or GT selection.

Selectively reimplements tag semantics of B models/fusion_v2.py::features
(data_dir, cache=None), commit a54219dbff8948507b112234127816c133f7c08b,
MIT (see repository LICENSE). All numerical model parameters are excluded.
R9 restart_proposals supplies the existing bounded restart contract.
"""
import math
import numpy as np
from solution.event_evidence import normalized, restart_proposals, HALVES

ATTACK = frozenset({'attacking transition', 'build up', 'progression',
                   'ball in final third', 'ball in the box', 'players in the box',
                   'creating chances', 'long ball'})

DEFENSE = frozenset({'defending in defensive third', 'defending in middle third',
                     'defending in attacking third'})


def tactical_evidence(rows, *, fps, n_frames, half_blocks, team_names):
    """Team tags remain uncertain supports, never possession/D labels.

    Times are global video seconds with Python round(seconds*fps), [start,end).
    Defense flips the explicitly mapped defending team. Unknown Team stays None.
    All inferred semantics remain uncalibrated, including supplied confidence.
    """
    rows = list(rows)
    restart_rows, restart_indices = [], []
    for i, row in enumerate(rows):
        try:
            if str(normalized(row).get('code', '')).strip().lower() == 'set pieces':
                restart_rows.append(row); restart_indices.append(i)
        except (ValueError, TypeError):
            pass  # Main parser records a single rejection with the original index.
    restart = restart_proposals(restart_rows, fps=fps, n_frames=n_frames,
                                half_blocks=half_blocks, team_names=team_names)
    for item in restart['rejected']:
        if 'row' in item: item['row'] = restart_indices[item['row']]
    names = normalized(team_names)
    if set(names.values()) != {'H', 'A'} or len(names) != 2:
        raise ValueError('explicit two-team mapping required')
    intervals, rejected, seen = [], list(restart['rejected']), set()
    for i, raw in enumerate(rows):
        try:
            r = normalized(raw); tags = normalized(r.get('tags', {}))
            code = str(r.get('code', '')).strip().lower()
            if code == 'set pieces': continue
            defending = code in DEFENSE
            if code not in ATTACK and not defending:
                raise ValueError('unsupported tactical category')
            half = HALVES.get(str(tags.get('half', '')).strip().lower())
            if half not in half_blocks: raise ValueError('unknown half')
            lo, hi = half_blocks[half]
            a, b = float(r['start_s']), float(r['end_s'])
            if not math.isfinite(a) or not math.isfinite(b) or b <= a:
                raise ValueError('invalid interval')
            a, b = round(a*fps), round(b*fps)
            if not lo <= a < b <= hi: raise ValueError('interval outside stated half')
            name = str(tags.get('team', '')).strip().lower()
            side = names.get(name)
            if side is None and name not in ('', 'n/a', 'none', 'unknown'):
                raise ValueError('unmapped team')
            if defending and side is not None: side = 'A' if side == 'H' else 'H'
            confidence = float(r.get('confidence', 1.))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('invalid confidence')
            source = str(r.get('source', 'automatic_tactical_unverified')).strip()
            if not source: raise ValueError('empty source')
            key = (half, a, b, code, side, confidence, source)
            if key in seen: continue
            seen.add(key)
            intervals.append(dict(start_frame=a, end_frame=b, half=half, team=side,
                type=code, source=source, confidence=confidence, unknown=side is None,
                confidence_status='UNCALIBRATED', hard_possession_label=False))
        except (ValueError, KeyError, TypeError) as e:
            rejected.append(dict(row=i, reason=str(e)))
    return dict(intervals=intervals, restart_proposals=restart['proposals'],
                rejected=rejected, full_phase_labels_produced=False,
                status='INTERFACE_READY_NOT_PERFORMANCE_VALIDATED')


def soft_team_support(evidence, *, frame, period):
    """Max support avoids repeated tags increasing weight; conflicts abstain."""
    frame, period = np.asarray(frame), np.asarray(period)
    if frame.ndim != 1 or period.shape != frame.shape:
        raise ValueError('unaligned timeline')
    support = np.zeros((len(frame), 2), float)
    for e in evidence['intervals']:
        if e['team'] is None: continue
        mask = (period == e['half']) & (frame >= e['start_frame']) & (frame < e['end_frame'])
        col = 0 if e['team'] == 'H' else 1
        support[mask, col] = np.maximum(support[mask, col], e['confidence'])
    conflict = np.all(support > 0, axis=1)
    support[conflict] = 0.
    return dict(home_minus_away=support[:, 0]-support[:, 1], conflict=conflict)
