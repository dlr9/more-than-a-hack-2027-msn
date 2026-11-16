"""Weak supervision from PUBLIC OTHER-match SkillCorner dynamic events.

This is a label adapter, never an inference feature producer. Event-derived
coordinates, pressure, EPV, phase, possession and outcome fields must not be
passed as predictor features for these labels. No absent event becomes D or a
negative. Inclusive frame_end is retained descriptively; only start anchors are
labelled, including the provider's one-frame (duration=0) possessions.
"""
from collections import Counter
import math
import re

import numpy as np


def _integer(value):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError('nonnegative integer expected')
    return int(value)


def _flag(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    if value is None or value == '':
        return None
    raise ValueError('invalid tri-state flag')


def adapt_skillcorner_events(rows, metadata, provenance):
    """Return weak start-anchor labels plus an audit, without opening any file.

    acquisition=1: provider-labelled interception/recovery, first possession of
    a new team, preceding matched possession by the opponent in the same half,
    nonoverlapping times and no explicit interruption on either side.
    acquisition=0: explicit completed same-team pass to this player followed by
    a matched reception. These are actual event negatives, not missing labels.
    Everything else remains -1. No gap is assumed continuously alive.
    Confidence is categorical source quality, never a calibrated probability.
    """
    required = {'source_kind', 'revision', 'sha256', 'match_id'}
    if not required <= provenance.keys() or provenance['source_kind'] != 'public_other_match':
        raise ValueError('explicit public OTHER-match provenance required')
    if not re.fullmatch(r'[a-f0-9]{64}', provenance['sha256']):
        raise ValueError('source SHA256 required')
    match_id = _integer(metadata['id'])
    if _integer(provenance['match_id']) != match_id:
        raise ValueError('provenance match mismatch')
    home, away = (str(metadata[k]['id']) for k in ('home_team', 'away_team'))
    if home == away:
        raise ValueError('duplicate teams')
    team_map = {home: 0, away: 1}
    player_team = {}
    for p in metadata['players']:
        key = str(p['id'])
        if key in player_team:
            raise ValueError('duplicate player identity')
        player_team[key] = str(p['team_id'])
    periods = {}
    for p in metadata['match_periods']:
        k = _integer(p['period'])
        a, b = _integer(p['start_frame']), _integer(p['end_frame'])
        if k not in (1, 2) or k in periods or b < a:
            raise ValueError('invalid period bounds')
        periods[k] = a, b
    out, seen, source_rows = [], set(), Counter()
    for row in rows:
        source_rows[row['event_type']] += 1
        if row['event_type'] != 'player_possession':
            continue
        if _integer(row['match_id']) != match_id:
            raise ValueError('mixed match')
        event_id = str(row['event_id'])
        if event_id in seen:
            raise ValueError('duplicate event')
        seen.add(event_id)
        period, start, end = (_integer(row[k]) for k in ('period', 'frame_start', 'frame_end'))
        if period not in periods or not periods[period][0] <= start <= end <= periods[period][1]:
            raise ValueError('event outside half')
        duration = float(row['duration'])
        if not math.isfinite(duration) or abs(duration - (end-start)/10.) > 1e-6:
            raise ValueError('10 FPS event duration mismatch')
        actor, team = str(row['player_id']), str(row['team_id'])
        if team not in team_map or player_team.get(actor) != team:
            raise ValueError('actor team disagrees with metadata')
        start_matched = _flag(row['is_player_possession_start_matched'])
        extrapolated = _flag(row['fully_extrapolated'])
        end_matched = _flag(row['is_player_possession_end_matched'])
        eligible = start_matched is True and extrapolated is False
        out.append(dict(event_id=event_id, frame=start, end_frame_inclusive=end,
            period=period, team=team_map[team], player_id=actor,
            event_kind=row['start_type'], contact_label=1 if eligible else -1,
            acquisition_label=-1, quality='matched_nonfully_extrapolated' if eligible else 'unknown',
            eligible=eligible, confidence='weak_provider_label_not_calibrated',
            source_match=match_id, source_revision=provenance['revision'],
            source_sha256=provenance['sha256'],
            _end_matched=end_matched, _first=_flag(row['first_player_possession_in_team_possession']),
            _before=row.get('game_interruption_before', ''),
            _after=row.get('game_interruption_after', ''),
            _end_type=row.get('end_type', ''), _pass_outcome=row.get('pass_outcome', ''),
            _target=str(row.get('player_targeted_id', '')),
            _previous_pass_matched=_flag(row.get('is_previous_pass_matched'))))
    out.sort(key=lambda e: (e['period'], e['frame'], e['event_id']))
    if len({(e['period'], e['frame']) for e in out}) != len(out):
        raise ValueError('ambiguous simultaneous possession starts')
    for previous, current in zip(out, out[1:]):
        linked = (previous['period'] == current['period'] and previous['eligible']
            and current['eligible'] and previous['_end_matched'] is True
            and previous['end_frame_inclusive'] < current['frame']
            and not previous['_after'] and not current['_before'])
        if not linked:
            continue
        if (previous['team'] != current['team'] and current['_first'] is True
            and current['event_kind'] in {'pass_interception', 'recovery'}):
            current['acquisition_label'] = 1
        elif (previous['team'] == current['team'] and current['_first'] is False
            and current['event_kind'] == 'pass_reception'
            and previous['_end_type'] == 'pass' and previous['_pass_outcome'] == 'successful'
            and previous['_target'] == current['player_id']
            and current['_previous_pass_matched'] is True):
            current['acquisition_label'] = 0
    for event in out:
        for key in list(event):
            if key.startswith('_'):
                del event[key]
    audit = dict(source_rows=dict(source_rows), player_possessions=len(out),
        eligible=sum(e['eligible'] for e in out),
        acquisition_counts=dict(Counter(str(e['acquisition_label']) for e in out)),
        event_kind_counts=dict(Counter(e['event_kind'] for e in out)),
        one_frame=sum(e['frame'] == e['end_frame_inclusive'] for e in out),
        phase_complement_used=False, missing_event_is_negative=False,
        full_spell_f1=None, validation_status='PARTIAL_WEAK_SUPERVISION_ONLY')
    return out, audit


def anchor_labels(events, frame, period):
    """Map exact CSV frame IDs to tracking rows; all other rows remain unknown."""
    frame, period = np.asarray(frame), np.asarray(period)
    if frame.ndim != 1 or frame.shape != period.shape or len(np.unique(frame)) != len(frame):
        raise ValueError('unique aligned tracking frames required')
    index = {int(f): i for i, f in enumerate(frame)}
    team = np.full(len(frame), -1, dtype=np.int8)
    acquisition = np.full(len(frame), -1, dtype=np.int8)
    for e in events:
        if e['frame'] not in index or period[index[e['frame']]] != e['period']:
            raise ValueError('event-to-tracking frame/half mismatch')
        i = index[e['frame']]
        if e['eligible']:
            if team[i] != -1:
                raise ValueError('duplicate anchor')
            team[i] = e['team']
            acquisition[i] = e['acquisition_label']
    return dict(team=team, acquisition=acquisition)
