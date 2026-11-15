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

def build_advanced_model(feats_path, data_dir=None):
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

    # 1. Short gap interpolation for missing ball (< 50 frames)
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

    # 2. Tactical tag priors
    tag_prior_H = np.zeros(n_frames, dtype=np.float32)
    tag_prior_A = np.zeros(n_frames, dtype=np.float32)
    dead_mask = np.zeros(n_frames, dtype=bool)
    dead_mask[H2_START:H2_KICKOFF] = True

    out_of_bounds = (ball_x < -1.0) | (ball_x > 106.3) | (ball_y < -1.0) | (ball_y > 69.0)
    dead_mask |= out_of_bounds

    att_codes = {'attacking transition', 'build up', 'progression', 'ball in final third', 'ball in the box', 'players in the box', 'creating chances', 'long ball'}
    def_codes = {'defending in defensive third', 'defending in middle third', 'defending in attacking third'}

    try:
        events = load_smart_tagging(data_dir=data_dir)
        for ev in events:
            code = ev['code'].strip().casefold()
            team = (ev['tags'].get('Team') or '').strip().casefold()
            sf = max(0, ev['start_frame'])
            ef = min(n_frames, ev['end_frame'])

            if code == 'set pieces':
                mid = (sf + ef) // 2
                dead_mask[sf:mid] = True
            elif code in att_codes:
                if 'barcelona' in team:
                    tag_prior_H[sf:ef] += 1.0
                elif 'atletic' in team:
                    tag_prior_A[sf:ef] += 1.0
            elif code in def_codes:
                if 'barcelona' in team:
                    tag_prior_A[sf:ef] += 1.0
                elif 'atletic' in team:
                    tag_prior_H[sf:ef] += 1.0
    except Exception as e:
        print("Warning loading smart tagging:", e)

    # 3. Rolling idle speed detection for dead ball
    speed_cumsum = np.concatenate(([0], np.cumsum(mean_speed)))
    win = 25
    idx = np.arange(n_frames)
    low_idx = np.maximum(0, idx - win)
    high_idx = np.minimum(n_frames, idx + win + 1)
    rolling_speed = (speed_cumsum[high_idx] - speed_cumsum[low_idx]) / (high_idx - low_idx)

    idle_dead = (rolling_speed < 1.55) & (is_missing | out_of_bounds)
    dead_mask |= idle_dead

    # Clean short dead ball blips (< 45 frames)
    d_diff = np.diff(np.pad(dead_mask.astype(int), (1, 1), 'constant'))
    d_starts = np.where(d_diff == 1)[0]
    d_ends = np.where(d_diff == -1)[0]
    for s, e in zip(d_starts, d_ends):
        if (e - s) < 45:
            dead_mask[s:e] = False

    # 4. State machine with prior-modulated sustained control
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
            pH = tag_prior_H[f]
            pA = tag_prior_A[f]

            if dH < 900 and dA < 900:
                diff = dA - dH

                # Prior boost: if tag strongly indicates a team, reduce distance threshold by up to 0.5m
                prior_diff = (pA - pH) * 0.25
                effective_diff = diff - prior_diff

                if current_state == 'H':
                    # To switch to A
                    req_sustain = 10 if pA > pH else 14
                    if effective_diff < -1.0 and dA < 3.2:
                        consecutive_opp += 1
                        if consecutive_opp >= req_sustain:
                            current_state = 'A'
                            consecutive_opp = 0
                    else:
                        consecutive_opp = max(0, consecutive_opp - 1)
                elif current_state == 'A':
                    # To switch to H
                    req_sustain = 10 if pH > pA else 14
                    if effective_diff > 1.0 and dH < 3.2:
                        consecutive_opp += 1
                        if consecutive_opp >= req_sustain:
                            current_state = 'H'
                            consecutive_opp = 0
                    else:
                        consecutive_opp = max(0, consecutive_opp - 1)
                else: # was D
                    if effective_diff > 0.3:
                        current_state = 'H'
                    elif effective_diff < -0.3:
                        current_state = 'A'
                    else:
                        current_state = 'H'

            half_labels.append(current_state)

        labels[half_start:half_end + 1] = half_labels

    # 5. Clean spurious micro-possessions (< 50 frames, 2.0s)
    def clean_spurious(arr, half_start, half_end, min_len=50):
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

    clean_spurious(labels, H1_START, H1_END, min_len=50)
    clean_spurious(labels, H2_START, H2_END, min_len=50)

    return labels

if __name__ == '__main__':
    feats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extracted_features.npz')
    gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation', 'gt', 'BAR-ATH_possession_gt_public.csv')
    gt = pm.label_codes(pm.read_label_csv(gt_path))

    labels = build_advanced_model(feats_path)
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation', 'results', 'model_fusion_v1.csv')
    df = pd.DataFrame({'frame': np.arange(len(labels)), 'label': labels})
    df.to_csv(out_csv, index=False)
    print("Saved submission to:", out_csv)

    pred = pm.label_codes(df)
    res = pm.evaluate(gt, pred)

    print("\n=== Model Fusion V1 Evaluation ===")
    print(f"  macroF1:   {res['macro_f1']:.4f}")
    print(f"  spellF1:   {res['spell_f1_mean']:.4f}")
    print(f"  possessions: {res['n_pred_spells']} (GT: {res['n_gt_spells']})")
    print(f"  mean dur:    {res['mean_spell_duration_s']:.1f} s (GT: {res['gt_mean_spell_duration_s']:.1f} s)")
    print(f"  kappa:       {res['kappa']:.4f}")
    print(f"  accuracy:    {res['accuracy']:.4f}")
