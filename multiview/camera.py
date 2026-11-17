"""Homography motion proxies, not a 3-D camera solution or ball detector."""
import numpy as np

def project(H,xy):
    H=np.asarray(H,float);xy=np.asarray(xy,float)
    if H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.cond(H)>1e10:raise ValueError('invalid homography')
    q=np.c_[xy,np.ones(len(xy))]@H.T
    if np.any(np.abs(q[:,2])<1e-9):raise ValueError('projective horizon')
    return q[:,:2]/q[:,2,None]

def camera_proxies(H,valid,error,frame,period,shot,fps,width,height,max_error_px=2.):
    from solution.observation_contract import validate_timeline
    contiguous=validate_timeline(frame,period,fps);H=np.asarray(H);n=len(H)
    if H.shape!=(n,3,3) or n!=len(frame) or len(valid)!=n or len(error)!=n or len(shot)!=n or width<=0 or height<=0:raise ValueError('camera contract')
    center=np.full((n,2),np.nan);pan=np.full((n,2),np.nan);zoom=np.full(n,np.nan);cut=np.zeros(n,bool);trusted=np.zeros(n,bool)
    grid=np.array([[.1,.1],[.5,.1],[.9,.1],[.1,.9],[.5,.9],[.9,.9]]);previous=None;scale=None
    for i in range(n):
        reset=i==0 or not contiguous[i-1] or shot[i]!=shot[i-1]
        if reset:previous=None;scale=None
        if not valid[i] or not np.isfinite(error[i]) or error[i]<0 or error[i]>max_error_px:previous=None;scale=None;continue
        try:
            points=project(H[i],grid);c=project(np.linalg.inv(H[i]),np.array([[width/2,height/2]]))[0]
        except (ValueError,np.linalg.LinAlgError):previous=None;scale=None;continue
        center[i]=c;trusted[i]=True;s=float(np.linalg.norm(points[2]-points[0]))
        if previous is not None and scale>0:
            shift=np.median(points-previous,axis=0);ratio=s/scale
            cut[i]=np.linalg.norm(shift)>.15*width or not .77<ratio<1.3
            if not cut[i]:pan[i]=shift*fps;zoom[i]=(ratio-1)*fps
        previous=points;scale=s
    velocity=np.full_like(center,np.nan)
    ok=contiguous&trusted[1:]&trusted[:-1]&~cut[1:]&(np.asarray(shot)[1:]==np.asarray(shot)[:-1])
    velocity[1:]=np.where(ok[:,None],np.diff(center,axis=0)*fps,np.nan)
    return dict(view_center_pitch=center,camera_pan_proxy_pxps=pan,camera_zoom_proxy_per_s=zoom,view_center_velocity=velocity,cut_suspected=cut,valid=trusted,independent_ball_measurement=False)
