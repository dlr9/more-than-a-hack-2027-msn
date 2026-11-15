import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'starter'))

import games
import baseline

def extract_features(raw_path, meta_path, n_frames=199319, pitch=(105.3, 68.0)):
    team_of_track = baseline.team_of_each_track(meta_path)
    pitch_len, pitch_wid = pitch

    min_dH = np.full(n_frames, 999.0, dtype=np.float32)
    min_dA = np.full(n_frames, 999.0, dtype=np.float32)
    ball_pres = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed = np.zeros(n_frames, dtype=np.float32)

    print("Extracting features from ATD...")
    with open(raw_path) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid >= n_frames:
                continue

            # Player coordinates and speeds
            p_speeds = []
            p_coords = []
            for slot, p_str in enumerate(parts[1].split(';')):
                if p_str[0] == 'N':
                    continue
                c1 = p_str.find(',')
                c2 = p_str.find(',', c1 + 1)
                px = float(p_str[:c1]) * pitch_len
                py = float(p_str[c1+1:c2]) * pitch_wid
                s_str = p_str[c2+1:]
                if s_str[0] != 'N':
                    try:
                        s_val = float(s_str)
                        if not np.isnan(s_val):
                            p_speeds.append(s_val)
                    except:
                        pass
                p_coords.append((team_of_track[slot], px, py))

            if p_speeds:
                mean_speed[fid] = sum(p_speeds) / len(p_speeds)

            b_str = parts[2]
            if b_str[0] != 'N':
                ball_pres[fid] = True
                c1 = b_str.find(',')
                c2 = b_str.find(',', c1 + 1)
                bx = float(b_str[:c1]) * pitch_len
                by = float(b_str[c1+1:c2]) * pitch_wid
                ball_x[fid] = bx
                ball_y[fid] = by

                best_H, best_A = 9999.0, 9999.0
                for tm, px, py in p_coords:
                    dist_sq = (px - bx)**2 + (py - by)**2
                    if tm == 0:
                        if dist_sq < best_H:
                            best_H = dist_sq
                    else:
                        if dist_sq < best_A:
                            best_A = dist_sq
                min_dH[fid] = np.sqrt(best_H)
                min_dA[fid] = np.sqrt(best_A)

    return {
        "min_dH": min_dH,
        "min_dA": min_dA,
        "ball_pres": ball_pres,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "mean_speed": mean_speed
    }

if __name__ == '__main__':
    raw_path, meta_path = games.require(['atd_raw', 'atd_meta'])
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extracted_features.npz')
    feats = extract_features(raw_path, meta_path)
    np.savez_compressed(out_path, **feats)
    print("Saved extracted_features.npz to:", out_path)
