import unittest
import numpy as np
from r10.impulse import impulse_strength

class ImpulseTests(unittest.TestCase):
    def raw(self):
        n=120;ball=np.column_stack([np.arange(n)*.4,np.zeros(n)]);players=np.repeat(ball[:,None,:],16,axis=1);players[:,8:,1]=10
        return dict(frame=np.arange(n),period=np.ones(n),ball_xy=ball,players_xy=players,player_team=np.r_[np.zeros(8,int),np.ones(8,int)],player_track_id=np.arange(16))
    def test_straight_motion_no_contact(self):
        s,v=impulse_strength(self.raw(),25);self.assertFalse(v.any());self.assertFalse(s.any())
    def test_stop_contact_team_swap_and_ids(self):
        r=self.raw();r['ball_xy'][60:]=r['ball_xy'][60];s,v=impulse_strength(r,25);self.assertGreater(s[60,0],s[60,1]);self.assertGreater(v[60],0)
        ix=np.arange(16)[::-1];z={**r,'players_xy':r['players_xy'][:,ix],'player_team':1-r['player_team'][ix],'player_track_id':np.arange(16)+10}
        np.testing.assert_array_equal(impulse_strength(z,25)[0],s[:,::-1])
    def test_missing_and_gap_are_not_impulses(self):
        r=self.raw();r['ball_xy'][60:]=r['ball_xy'][60];r['period'][60:]=2
        self.assertFalse(impulse_strength(r,25)[1].any());r['period'][:]=1;r['frame'][60:]+=10
        self.assertFalse(impulse_strength(r,25)[1].any());r['frame']=np.arange(120);r['ball_xy'][57:64]=np.nan
        self.assertFalse(impulse_strength(r,25)[1].any())
if __name__=='__main__':unittest.main()
