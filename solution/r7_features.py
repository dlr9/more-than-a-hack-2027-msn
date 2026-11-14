"""Versioned R7 ablations and label-blind interval corruption of raw positions."""
import hashlib
import json
import numpy as np
from .candidate import Candidate
from .provider_neutral import FeatureContract, assert_compatible

DIRECT = {'ball_speed_mps', 'ball_speed_missing', 'ball_displacement_0p2s_m',
          'ball_displacement_0p5s_m', 'ball_displacement_1p0s_m'}
GEOMETRY_EXTRA = {f'{team}_local_speed_mean_mps' for team in ('home', 'away')} | {
    f'{team}_d1_mean_{window}s' for team in ('home', 'away') for window in ('0p2', '0p5', '1p0')}
VIEWS = {'no_ball_motion': DIRECT, 'geometry_context': DIRECT | GEOMETRY_EXTRA}


def feature_hash(names):
    return hashlib.sha256(json.dumps(list(names), separators=(',', ':')).encode()).hexdigest()


def project(batch, view):
    if view not in VIEWS or int(batch['schema_version']) != 3:
        raise ValueError('R7 projection requires named view and schema3 base')
    names = batch['feature_names'].tolist()
    if not VIEWS[view].issubset(names):
        raise ValueError('missing expected source columns')
    keep = [i for i, name in enumerate(names) if name not in VIEWS[view]]
    names = [names[i] for i in keep]
    base = str(batch['feature_contract_json'])
    FeatureContract(**json.loads(base))
    contract = dict(schema_version=4, definition='R7_'+view, feature_names_sha256=feature_hash(names), base=json.loads(base))
    return {**batch, 'X': batch['X'][:, keep].copy(), 'feature_names': np.asarray(names),
            'schema_version': np.asarray(4), 'feature_view': np.asarray(view),
            'base_feature_contract_json': np.asarray(base),
            'feature_contract_json': np.asarray(json.dumps(contract, sort_keys=True))}


class AblatedCandidate(Candidate):
    def probabilities(self, batch):
        view = self.metadata['feature_view']
        if str(batch.get('feature_view', '')) != view:
            raise ValueError('R7 feature view mismatch')
        definition = json.loads(str(batch['feature_contract_json']))
        names = batch['feature_names'].tolist()
        if definition['definition'] != 'R7_'+view or definition['feature_names_sha256'] != feature_hash(names):
            raise ValueError('R7 feature contract mismatch')
        base = FeatureContract(**definition['base'])
        sources = [FeatureContract(**c) for c in self.metadata['training_contracts']]
        assert_compatible(sources, base, purpose='evaluate', allow_cross_pitch=True)
        return super().probabilities(batch)


def blind_intervals(period, fps, *, scheme, seed):
    """Only period/fps/seed enter the mask; no coordinates, labels or predictions."""
    if scheme not in {'short', 'long'} or not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid interval process')
    period = np.asarray(period)
    mask = np.zeros(len(period), dtype=bool)
    windows = []
    from .quality import period_blocks
    for block_index, (a, b) in enumerate(period_blocks(period)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, block_index]))
        durations, gaps = ((.2, .5, 1.), (2., 6.)) if scheme == 'short' else ((2., 5., 10.), (20., 60.))
        cursor = a
        while cursor <= b:
            cursor += max(1, int(round(rng.uniform(*gaps)*fps)))
            if cursor > b:
                break
            end = min(b+1, cursor+max(1, int(round(rng.choice(durations)*fps))))
            mask[cursor:end] = True
            windows.append((cursor, end, int(period[cursor])))
            cursor = end
    return mask, windows
