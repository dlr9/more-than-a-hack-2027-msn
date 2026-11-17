from dataclasses import dataclass
import numpy as np
from solution.observation_contract import validate_timeline

@dataclass
class View:
    source: str
    fps: float
    frame: np.ndarray
    period: np.ndarray
    time_s: np.ndarray
    players: np.ndarray  # normalized fixed pitch axes, NxPx2; NaN=unobserved
    teams: np.ndarray    # NxP, -1 unknown, 0/1 canonical teams
    ball: np.ndarray     # Nx2; raw source observations, not interpolation
    ball_observed: np.ndarray
    shot: np.ndarray
    replay: np.ndarray
    homography_valid: np.ndarray
    reprojection_error: np.ndarray
    clock_s: np.ndarray | None=None  # independent match clock, NOT video time
    ball_grounded: np.ndarray | None=None
    pitch_m: tuple=(105.,68.)
    def __post_init__(self):
        n=len(self.frame)
        if not self.source or not np.isfinite(self.fps) or self.fps<=0:raise ValueError('source/fps')
        validate_timeline(self.frame,self.period,self.fps,self.time_s)
        if self.players.ndim!=3 or self.players.shape[0]!=n or self.players.shape[2]!=2 or self.teams.shape!=self.players.shape[:2]:raise ValueError('player shape')
        if self.ball.shape!=(n,2):raise ValueError('ball shape')
        for k in ['period','time_s','ball_observed','shot','replay','homography_valid','reprojection_error']:
            if np.asarray(getattr(self,k)).shape!=(n,):raise ValueError(k)
        for k in ['ball_observed','replay','homography_valid']:
            if np.asarray(getattr(self,k)).dtype!=bool:raise ValueError(k+' must be boolean')
        if np.isinf(self.players).any() or np.isinf(self.ball).any():raise ValueError('infinite coordinates')
        if not np.isin(self.teams,[-1,0,1]).all():raise ValueError('team mapping')
        if np.any(self.ball_observed&~np.isfinite(self.ball).all(1)):raise ValueError('observed ball must exist')
        for k in ['clock_s','ball_grounded']:
            v=getattr(self,k)
            if v is not None and np.asarray(v).shape!=(n,):raise ValueError(k)
        if self.ball_grounded is not None and np.asarray(self.ball_grounded).dtype!=bool:raise ValueError('ground mask')
        if len(self.pitch_m)!=2 or not np.isfinite(self.pitch_m).all() or min(self.pitch_m)<=0:raise ValueError('pitch')
        finite=self.players[np.isfinite(self.players).all(-1)]
        if len(finite) and np.max(np.abs(finite))>2:raise ValueError('normalized coordinates required')
