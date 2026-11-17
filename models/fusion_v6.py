"""V6: V5 physical/kinematic decoding plus Smart Tagging restart-head latch.

Promotes the independently evaluated tag_head experiment without retuning.
This infers dead-ball preparation, not a verified set-piece subtype or kick instant.
"""
import fusion_v5 as v5
from restart_experiments import masks

DEFAULT = {k: v for k, v in v5.DEFAULT.items() if k != 'restart_experiment'}


def features(data_dir=None):
    return v5.features(data_dir)


def predict_with_audit(f, c=None, prepared=None):
    cfg = dict(DEFAULT, **(c or {}))
    if set(cfg) != set(DEFAULT):
        raise ValueError('V6 configuration contains unsupported keys')
    if 'restart_events' not in f:
        raise ValueError('V6 requires Smart Tagging events; load with fusion_v6.features')
    labels = v5.predict(f, dict(cfg, restart_experiment='none'), prepared=prepared)
    held, audit = masks(f, v5.BLOCKS, f['restart_events'], 'tag_head')
    labels = labels.copy()
    labels[held] = 'D'
    return labels, audit


def predict(f, c=None, prepared=None):
    return predict_with_audit(f, c, prepared)[0]
