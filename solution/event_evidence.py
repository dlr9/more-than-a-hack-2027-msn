"""Generic automatic restart evidence interface; synthetic-tested only.

Accepts already parsed public-contract rows, never opens files. Returns bounded
uncertain boundary proposals, never a phase-wide team label or a D complement.
Target use is exclusively an authorized human's isolated offline workflow.
"""
import math

TYPES={'throw-in','corner kick','goal kick','free kick','kick off (after goal)','kick off (start)'}
HALVES={'1st half':1,'2nd half':2}

def normalized(mapping):
    result={}
    for key,value in mapping.items():
        k=str(key).strip().lower()
        if k in result:raise ValueError('duplicate case-insensitive field')
        result[k]=value
    return result

def restart_proposals(rows, *, fps, n_frames, half_blocks, team_names):
    """half_blocks maps half to [start,stop); team_names maps names to H/A.

    Start time is only a soft candidate, not an observed restart instant. The
    phase end checks validity but is never expanded into possession labels.
    """
    if not math.isfinite(fps) or fps<=0 or n_frames<=0:raise ValueError('time contract')
    blocks=sorted((int(h),int(a),int(b)) for h,(a,b) in half_blocks.items())
    if any(not 0<=a<b<=n_frames for _,a,b in blocks):raise ValueError('half bounds')
    if any(a<z for (_,_,z),(_,a,_) in zip(blocks,blocks[1:])):raise ValueError('overlapping halves')
    names={str(k).strip().lower():v for k,v in team_names.items()}
    if any(v not in ('H','A') for v in names.values()):raise ValueError('team encoding')
    accepted=[];rejected=[];seen=set()
    for i,row in enumerate(rows):
        try:
            r=normalized(row);tags=normalized(r.get('tags',{}))
            if str(r.get('code','')).strip().lower()!='set pieces':continue
            kind=str(tags.get('type','')).strip().lower()
            if kind not in TYPES:raise ValueError('unknown restart type')
            half=HALVES.get(str(tags.get('half','')).strip().lower())
            if half not in half_blocks:raise ValueError('unknown half')
            lo,hi=half_blocks[half]
            start=float(r['start_s']);end=float(r['end_s'])
            if not math.isfinite(start) or not math.isfinite(end) or end<=start:raise ValueError('invalid interval')
            a,b=round(start*fps),round(end*fps)
            if not lo<=a<b<=hi:raise ValueError('interval outside stated half')
            name=str(tags.get('team','')).strip().lower()
            team=names.get(name)
            if team is None and name not in ('','n/a','none','unknown'):raise ValueError('unmapped team')
            direction=str(tags.get('direction of ball entry','')).strip().lower()
            if direction and direction not in {'horizontal','vertical','diagonal'}:raise ValueError('unknown direction')
            key=(a,b,kind,half,team)
            if key in seen:continue
            seen.add(key);radius=max(1,round(.5*fps))
            accepted.append(dict(frame=a,window=[max(lo,a-radius),min(hi,a+radius+1)],team=team,half=half,
                type=kind,source='automatic_restart_interval',confidence='uncalibrated',status='INTERFACE_READY_NOT_PERFORMANCE_VALIDATED'))
        except (ValueError,KeyError,TypeError) as e:rejected.append(dict(row=i,reason=str(e)))
    # Conflicting team tags at one restart cannot be resolved from this interface.
    conflicts={e['frame'] for e in accepted if len({v['team'] for v in accepted if v['frame']==e['frame'] and v['team'] is not None})>1}
    kept=[e for e in accepted if e['frame'] not in conflicts]
    rejected.extend(dict(frame=f,reason='conflicting teams') for f in sorted(conflicts))
    return dict(proposals=kept,rejected=rejected,full_phase_labels_produced=False)
