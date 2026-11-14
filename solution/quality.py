"""External-only coverage audits. Absence is unknown, never proof of dead ball."""
import csv
import numpy as np

def period_blocks(period):
    cuts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return [(int(a), int(b-1)) for a,b in zip(cuts[:-1],cuts[1:]) if b>a and period[a]>0]

def dropout_windows(period, fps, seed=2027):
    mask = np.zeros(len(period), dtype=bool)
    rng = np.random.default_rng(seed)
    for a,b in period_blocks(period):
        for start in range(a, b+1, round(20*fps)):
            length = round(float(rng.choice([.5,2.,5.]))*fps)
            offset = int(rng.integers(0, max(1, round(10*fps))))
            mask[min(b+1,start+offset):min(b+1,start+offset+length)] = True
    return mask

def phase_coverage(batch, events_path):
    names = batch['feature_names'].astype(str).tolist()
    visible = batch['X'][:,names.index('ball_visible')] == 1
    inside = batch['period']>0
    gap = inside & (batch['raw_y_alive']==0)
    live = np.zeros(len(gap),dtype=bool)
    with open(events_path, encoding='utf-8-sig') as f:
        for event in csv.DictReader(f):
            # All exported dynamic actions count as activity evidence, not independent GT.
            a,b=int(event['frame_start']),int(event['frame_end'])
            live[max(0,a):min(len(live),b)] = True
    gaps=[]
    interior_overlap=0
    for a,b in period_blocks(batch['period']):
        padded=np.r_[False,gap[a:b+1],False].astype(int)
        starts=np.flatnonzero(np.diff(padded)==1)+a
        ends=np.flatnonzero(np.diff(padded)==-1)+a
        for lo,hi in zip(starts,ends):
            margin=round(float(batch["fps"]))
            if hi-lo>2*margin:
                interior_overlap += int(live[lo+margin:hi-margin].sum())
            gaps.append(dict(period=int(batch['period'][lo]),seconds=float((hi-lo)/batch['fps']),ball_visible_fraction=float(visible[lo:hi].mean()),dynamic_activity_frames=int(live[lo:hi].sum())))
    known=batch['y_home']>=0
    return dict(match_id=str(batch['match_id']), in_period_frames=int(inside.sum()),phase_gap_frames=int(gap.sum()),gap_activity_overlap=int((gap&live).sum()),gap_interior_activity_overlap=interior_overlap,ha_frames=int(known.sum()),ha_missing_ball_frames=int((known&~visible).sum()),detected_players=int(batch['player_quality_counts'][:,0].sum()),extrapolated_players=int(batch['player_quality_counts'][:,1].sum()),gaps=gaps,label_semantics='unverified_weak',trusted_dead_frames=0)
