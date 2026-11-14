"""Generated DFL XML only: object-major alignment, labels and malformed intervals."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from solution.idsse import read_raw,label

class IDSSETests(unittest.TestCase):
    def fixture(self,root,duplicate=False,badtime=False):
        meta=root/'meta.xml';pos=root/'positions.xml'
        meta.write_text('<Root><General MatchId="DFL-MAT-SYNTH" HomeTeamId="h" GuestTeamId="a"/>'
                        '<Environment PitchX="100" PitchY="60"/><Teams>'
                        '<Team Role="home" TeamId="h"><Players><Player PersonId="ph"/></Players></Team>'
                        '<Team Role="guest" TeamId="a"><Players><Player PersonId="pa"/></Players></Team></Teams></Root>')
        sections=[]
        for k,lo in [('firstHalf',10),('secondHalf',100)]:
            for team,person,x in [('h','ph',-10),('a','pa',10),('ball','ball',0)]:
                rows=[]
                for j in range(4):
                    ms=j*40+(1 if badtime and j==2 else 0)
                    attrs=f'N="{lo+j}" T="2022-01-01T00:00:00.{ms:03d}+00:00" X="{x+j}" Y="0" S="36"'
                    if team=='ball':attrs+=f' BallStatus="{[1,1,0,1][j]}" BallPossession="{[1,2,0,0][j]}"'
                    rows.append(f'<Frame {attrs}/>')
                if team=='ball'and duplicate:rows.append(rows[0])
                sections.append(f'<FrameSet GameSection="{k}" TeamId="{team}" PersonId="{person}" MatchId="DFL-MAT-SYNTH">'+''.join(rows)+'</FrameSet>')
        pos.write_text('<Root>'+''.join(sections)+'</Root>');return pos,meta

    def test_object_major_alignment_and_units(self):
        with tempfile.TemporaryDirectory()as d:
            p,m=self.fixture(Path(d));raw,y,meta,audit,extra=read_raw(p,m)
            self.assertEqual(''.join(y),'HADXHADX')
            np.testing.assert_array_equal(raw['period'],[1]*4+[2]*4)
            self.assertEqual(audit['missing_ball_records'],0)
            self.assertEqual(audit['known_fraction'],.75)
            self.assertEqual(extra['reported_speed_mps'][1,0],10.)
            self.assertEqual(raw['players_xy'][1,0,0],11.) # pa slot sorted before ph
            self.assertEqual(raw['players_xy'][1,1,0],-9.)
            self.assertEqual(audit['firstHalf']['longest_unknown_s'],.04)

    def test_conflicting_ball_record_becomes_unknown(self):
        with tempfile.TemporaryDirectory()as d:
            p,m=self.fixture(Path(d),duplicate=True);raw,y,_,audit,_=read_raw(p,m)
            self.assertEqual(audit['duplicate_ball_records'],2)
            self.assertEqual(y[0],'X');self.assertTrue(np.isnan(raw['ball_xy'][0]).all())

    def test_bad_timestamp_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            p,m=self.fixture(Path(d),badtime=True)
            with self.assertRaisesRegex(ValueError,'timestamp'):read_raw(p,m)

    def test_unknown_status_not_inferred_from_possession(self):
        for status,pos in [(None,'1'),('2','2'),('1','0'),('1',None)]:self.assertEqual(label(status,pos),'X')
        self.assertEqual(label('0',None),'D')

if __name__=='__main__':unittest.main()
