import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'starter'))

import games
import possession_metrics as pm
from load_smart_tagging import load as load_smart_tagging

def run_experiment(feats_path, sustain_frames=15, min_spell_len=45, min_dead_len=50, opp_dist_thr=3.2, dist_diff_thr=1.2):
    data = np.load(feats_path)
    min_dH = data['min_dH'].copy()
    min_dA = data['min_dA'].copy()
    ball_pres = data['ball_pres'].copy()
    ball_x = data['ball_x'].copy()
    ball_y = data['ball_y'].copy()
    mean_speed = data['mean_speed'].copy()

    n_frames = len(min_dH)
    labels = np.full(n_frames, 'X', dtype='<U1')

    H1_START, H1_END = 11185, 90651
    H2_START, H2_END = 110808, 190197
    H2_KICKOFF = 113742

    # 1. Short gap interpolation
    is_missing = ~ball_pres
    d = np.diff(np.pad(is_missing.astype(int), (1, 1), 'constant'))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        gap = e - s
        if gap <= 50 and s > 0 and e < n_frames:
            dH_start, dH_end = min_dH[s - 1], min_dH[e]
            dA_start, dA_end = min_dA[s - 1], min_dA[e]
            if dH_start < 900 and dH_end < 900:
                min_dH[s:e] = np.linspace(dH_start, dH_end, gap)
            if dA_start < 900 and dA_end < 900:
                min_dA[s:e] = np.linspace(dA_start, dA_end, gap)

    # 2. Dead ball mask
    dead_mask = np.zeros(n_frames, dtype=bool)
    dead_mask[H2_START:H2_KICKOFF] = True

    out_of_bounds = (ball_x < -1.0) | (ball_x > 106.3) | (ball_y < -1.0) | (ball_y > 69.0)
    dead_mask |= out_of_bounds

    try:
        events = load_smart_tagging(data_dir=r'F:\non_video_dataset')
        for ev in events:
            code = ev['code'].strip().casefold()
            sf = max(0, ev['start_frame'])
            ef = min(n_frames, ev['end_frame'])
            if code == 'set pieces':
                mid = (sf + ef) // 2
                dead_mask[sf:mid] = True
    except Exception as e:
        pass

    speed_cumsum = np.concatenate(([0], np.cumsum(mean_speed)))
    win = 25
    idx = np.arange(n_frames)
    low_idx = np.maximum(0, idx - win)
    high_idx = np.minimum(n_frames, idx + win + 1)
    rolling_speed = (speed_cumsum[high_idx] - speed_cumsum[low_idx]) / (high_idx - low_idx)

    idle_dead = (rolling_speed < 1.55) & (is_missing | out_of_bounds)
    dead_mask |= idle_dead

    # Clean short dead ball blips (< min_dead_len)
    def clean_short_dead(mask, min_len=min_dead_len):
        d_diff = np.diff(np.pad(mask.astype(int), (1, 1), 'constant'))
        d_starts = np.where(d_diff == 1)[0]
        d_ends = np.where(d_diff == -1)[0]
        for s, e in zip(d_starts, d_ends):
            if (e - s) < min_len:
                mask[s:e] = False

    clean_short_dead(dead_mask, min_dead_len)

    # 3. State machine
    for half_start, half_end in [(H1_START, H1_END), (H2_START, H2_END)]:
        h_score = np.sum(min_dH[half_start:half_start+50] < min_dA[half_start:half_start+50])
        a_score = np.sum(min_dA[half_start:half_start+50] < min_dH[half_start:half_start+50])
        current_state = 'H' if h_score >= a_score else 'A'

        half_labels = []
        consecutive_opp = 0

        for f in range(half_start, half_end + 1):
            if dead_mask[f]:
                half_labels.append('D')
                consecutive_opp = 0
                continue

            dH = min_dH[f]
            dA = min_dA[f]

            if dH < 900 and dA < 900:
                diff = dA - dH
                if current_state == 'H':
                    if dA < dH - dist_diff_thr and dA < opp_dist_thr:
                        consecutive_opp += 1
                        if consecutive_opp >= sustain_frames:
                            current_state = 'A'
                            consecutive_opp = 0
                    else:
                        consecutive_opp = max(0, consecutive_opp - 1)
                elif current_state == 'A':
                    if dH < dA - dist_diff_thr and dH < opp_dist_thr:
                        consecutive_opp += 1
                        if consecutive_opp >= sustain_frames:
                            current_state = 'H'
                            consecutive_opp = 0
                    else:
                        consecutive_opp = max(0, consecutive_opp - 1)
                else: # was D
                    if diff > 0.5:
                        current_state = 'H'
                    elif diff < -0.5:
                        current_state = 'A'
                    else:
                        current_state = 'H'

            half_labels.append(current_state)

        labels[half_start:half_end + 1] = half_labels

    # 4. Clean spurious possessions
    def clean_spurious(arr, half_start, half_end, min_len=min_spell_len):
        sub = arr[half_start:half_end + 1]
        change = np.where(sub[:-1] != sub[1:])[0]
        starts = np.r_[0, change + 1]
        ends = np.r_[change, len(sub) - 1]
        for i in range(len(starts)):
            length = ends[i] - starts[i] + 1
            curr_lbl = sub[starts[i]]
            if length < min_len and curr_lbl in ('H', 'A'):
                prev_lbl = sub[starts[i-1]] if i > 0 else None
                next_lbl = sub[starts[i+1]] if i < len(starts) - 1 else None
                if prev_lbl and prev_lbl == next_lbl:
                    sub[starts[i]:ends[i]+1] = prev_lbl
                elif prev_lbl and prev_lbl in ('H', 'A'):
                    sub[starts[i]:ends[i]+1] = prev_lbl
                elif next_lbl and next_lbl in ('H', 'A'):
                    sub[starts[i]:ends[i]+1] = next_lbl
        arr[half_start:half_end + 1] = sub

    clean_spurious(labels, H1_START, H1_END, min_len=min_spell_len)
    clean_spurious(labels, H2_START, H2_END, min_len=min_spell_len)

    return labels

if __name__ == '__main__':
    feats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extracted_features.npz')
    gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation', 'gt', 'BAR-ATH_possession_gt_public.csv')
    gt = pm.label_codes(pm.read_label_csv(gt_path))

    for sustain in [12, 16, 20]:
        for min_spell in [35, 50, 65]:
            for min_dead in [35, 50]:
                lbls = run_experiment(feats_path, sustain_frames=sustain, min_spell_len=min_spell, min_dead_len=min_dead)
                df = pd.DataFrame({'frame': np.arange(len(lbls)), 'label': lbls})
                pred = pm.label_codes(df)
                res = pm.evaluate(gt, pred)
                print(f"sustain={sustain:2d}, spell_len={min_spell:2d}, dead_len={min_dead:2d} -> "
                      f"spellF1: {res['spell_f1_mean']:.4f}, macroF1: {res['macro_f1']:.4f}, "
                      f"poss: {res['n_pred_spells']:2d} (GT 17), dur: {res['mean_spell_duration_s']:4.1f}s")
