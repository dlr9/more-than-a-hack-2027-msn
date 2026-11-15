"""Research-only observable-quality gating; no source label access."""
import numpy as np

def gate_mask(base,players):
    if not all(np.array_equal(base[k],players[k])for k in ['frame','period','fps']):
        raise ValueError('unaligned expert timelines or FPS')
    names=base['feature_names'].tolist();pn=players['feature_names'].tolist()
    visible=base['X'][:,names.index('ball_visible')]==1
    frame,period=base['frame'],base['period'];age=np.zeros(len(frame));fps=float(base['fps'])
    for i in range(len(frame)):
        if period[i]<=0 or visible[i]:continue
        continued=i>0 and period[i]==period[i-1] and frame[i]==frame[i-1]+1
        age[i]=(age[i-1] if continued else 0)+1/fps
    x=players['X']
    reliable=(x[:,pn.index('home_count')]>=8)&(x[:,pn.index('away_count')]>=8)&(x[:,pn.index('speed_observed_fraction')]>=.5)
    return (age>=.5)&reliable

def fuse(base,players,ball_alive,player_alive):
    use=gate_mask(base,players)
    return np.where(use,player_alive,ball_alive),use
