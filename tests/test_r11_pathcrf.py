import unittest
import numpy as np
from r11.pathcrf_adapter import adapt_window, ContractError


def fixture(fps=25):
    n = 10*fps
    xy = np.zeros((n, 22, 2)); ids = np.array([f'p{i:02}' for i in range(22)])
    xy[:, :11, 0] = np.linspace(-48, 10, 11)
    xy[:, 11:, 0] = np.linspace(48, -10, 11)
    xy[:, :, 1] = np.tile(np.linspace(-25, 25, 11), 2)
    xy[:, :, 0] += .2*np.arange(n)[:, None]/fps
    return dict(frame=np.arange(n), period=np.ones(n), players_xy=xy,
                player_team=np.repeat([0, 1], 11), player_track_id=ids,
                fps=fps, pitch_m=[105., 68.], goalkeeper_ids={0:'p00',1:'p11'})


class PathCRFContractTests(unittest.TestCase):
    def test_25_and_10hz_layout_units_and_role(self):
        for fps in (10, 25):
            x = adapt_window(**fixture(fps))
            self.assertEqual(x['nodes'].shape, (50, 26, 8))
            np.testing.assert_array_equal(np.diff(x['frames']), fps//5)
            np.testing.assert_allclose(x['nodes'][:, :22, 4], .2, atol=1e-5)
            np.testing.assert_array_equal(x['nodes'][:, :11, 0], -1)
            np.testing.assert_array_equal(x['nodes'][:, 11:22, 0], 1)
            self.assertEqual(x['nodes'][0, :22, 1].sum(), 2)

    def test_per_frame_slot_permutation_exact_identity(self):
        a = fixture(); expected = adapt_window(**a)
        rng = np.random.default_rng(1101); n = len(a['frame'])
        teams=np.broadcast_to(a['player_team'], (n,22)).copy()
        ids=np.broadcast_to(a['player_track_id'], (n,22)).copy()
        for i in range(n):
            order=rng.permutation(22);a['players_xy'][i]=a['players_xy'][i,order]
            teams[i]=teams[i,order];ids[i]=ids[i,order]
        a.update(player_team=teams,player_track_id=ids)
        actual = adapt_window(**a)
        np.testing.assert_array_equal(actual['nodes'], expected['nodes'])

    def test_complete_and_one_second_outage_rejected(self):
        for lo, hi in [(0,250),(40,65)]:
            a=fixture();a['players_xy'][lo:hi]=np.nan
            with self.assertRaises(ContractError):adapt_window(**a)

    def test_unknown_gk_duplicate_identity_or_changed_team_rejected(self):
        a=fixture();a['goalkeeper_ids']=None
        with self.assertRaises(ContractError):adapt_window(**a)
        a=fixture();a['player_track_id'][1]='p00'
        with self.assertRaises(ContractError):adapt_window(**a)
        a=fixture();a['player_team'][0]=1
        with self.assertRaises(ContractError):adapt_window(**a)

    def test_half_and_frame_gap_no_bridge(self):
        a=fixture();a['period'][100:]=2
        with self.assertRaises(ContractError):adapt_window(**a)
        a=fixture();a['frame'][100:]+=1
        with self.assertRaises(ContractError):adapt_window(**a)

    def test_team_swap_preserves_geometric_tensor_changes_mapping(self):
        a=fixture();before=adapt_window(**a)
        a['player_team']=1-a['player_team'];a['goalkeeper_ids']={0:'p11',1:'p00'}
        after=adapt_window(**a)
        np.testing.assert_array_equal(after['nodes'],before['nodes'])
        self.assertEqual(after['left_team'],1-before['left_team'])

    def test_mirror_has_correct_velocities_and_outside_nodes(self):
        a=fixture();a['players_xy'][:,:,0]*=-1
        b=adapt_window(**a)
        self.assertEqual(b['left_team'],1)
        np.testing.assert_allclose(b['nodes'][:,:22,4],-.2,atol=1e-5)
        np.testing.assert_array_equal(b['nodes'][0,22:,2:4],[[0,34],[105,34],[52.5,0],[52.5,68]])

    def test_no_ball_or_target_input_argument(self):
        with self.assertRaises(TypeError): adapt_window(**fixture(), ball_xy=np.zeros((250,2)))


if __name__=='__main__':unittest.main()
