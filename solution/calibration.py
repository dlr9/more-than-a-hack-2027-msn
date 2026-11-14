"""Calibration on separate whole matches, with explicit uncalibrated fallback."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class BinaryCalibration:
    def __init__(self):
        self.model = None
        self.status = 'not_fitted'

    @staticmethod
    def _logit(p):
        p = np.asarray(p, dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('invalid probabilities')
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y, *, training_ids, calibration_ids):
        if not calibration_ids or set(training_ids) & set(calibration_ids):
            raise ValueError('calibration must use disjoint whole matches')
        if set(np.unique(y)) != {0, 1}:
            self.status = 'unavailable_single_class'
            return self
        self.model = LogisticRegression(C=1, random_state=2027).fit(self._logit(p), y)
        self.status = 'fitted'
        return self

    def predict(self, p):
        x = self._logit(p)
        return np.asarray(p) if self.model is None else self.model.predict_proba(x)[:, 1]

def probability_metrics(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return dict(n=0, status='not_tested')
    BinaryCalibration._logit(p)
    reliability, ece = [], 0.
    index = np.minimum((p * bins).astype(int), bins - 1)
    for i in range(bins):
        mask = index == i
        if not mask.any():
            continue
        confidence, frequency = float(p[mask].mean()), float(y[mask].mean())
        ece += mask.mean() * abs(confidence - frequency)
        reliability.append(dict(bin=i, n=int(mask.sum()), predicted=confidence, observed=frequency))
    return dict(n=len(y), brier=float(np.mean((p-y)**2)), log_loss=float(log_loss(y,p,labels=[0,1])), ece=float(ece), reliability=reliability)
