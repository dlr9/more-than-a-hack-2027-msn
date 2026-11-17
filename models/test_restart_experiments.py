import unittest
import numpy as np
from restart_experiments import formation,masks,hull_area

class RestartTests(unittest.TestCase):
    def test_penalty_missing_and_hull(self):
        p=np.array([[0.,34.],[12.,34.]]+[[20.+i,15.+i] for i in range(20)])
        t=np.array([0,1]+[0]*10+[1]*10)
        self.assertEqual(formation(p,t)['type'],'penalty')
        p[2]=np.nan
        self.assertIsNone(formation(p,t))
        self.assertEqual(hull_area(np.array([[0,0],[1,0],[1,1],[0,1]])),1)

    def test_corner(self):
        p=np.array([[0.,0.]]+[[5.+i%5,22.+i] for i in range(15)]+[[50.+i,20.] for i in range(6)])
        self.assertEqual(formation(p,np.repeat([0,1],11))['type'],'corner')

    def test_tag_end_and_no_ball_bridge(self):
        f=dict(players=np.zeros((100,22,2)),ball=np.full((100,2),np.nan),
               teams=np.repeat([0,1],11),mean_speed=np.zeros(100))
        e=[dict(code='SET PIECES',start_frame=10,end_frame=60,tags={'Type':'Free Kick'})]
        held,audit=masks(f,[(0,100)],e,'tag_head')
        self.assertEqual(np.flatnonzero(held).tolist(),list(range(10,60)))
        self.assertEqual(audit[0]['release'],'tag_end')
        f['mean_speed'][:35]=2
        held,_=masks(f,[(0,100)],e,'tag_head')
        self.assertFalse(held.any())

if __name__=='__main__':unittest.main()
