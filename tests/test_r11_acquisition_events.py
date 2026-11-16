import copy
import unittest

import numpy as np

from r11.acquisition_events import adapt_skillcorner_events, anchor_labels


class AcquisitionEventTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(id=99, home_team={'id': 10}, away_team={'id': 20},
            players=[{'id': 7, 'team_id': 10}, {'id': 8, 'team_id': 10}, {'id': 9, 'team_id': 20}],
            match_periods=[dict(period=1, start_frame=10, end_frame=100),
                           dict(period=2, start_frame=110, end_frame=200)])
        self.prov = dict(source_kind='public_other_match', revision='frozen', sha256='a'*64, match_id=99)

    def row(self, frame=20, player=7, team=10, **extra):
        r = dict(event_id=str(frame), event_type='player_possession', match_id='99',
            frame_start=str(frame), frame_end=str(frame+2), duration='0.2', period='1',
            team_id=str(team), player_id=str(player), start_type='pass_reception',
            is_player_possession_start_matched='True', fully_extrapolated='False',
            is_player_possession_end_matched='True', first_player_possession_in_team_possession='False',
            end_type='pass', pass_outcome='successful', player_targeted_id='8',
            is_previous_pass_matched='True', game_interruption_before='', game_interruption_after='')
        return {**r, **extra}

    def adapt(self, rows):
        return adapt_skillcorner_events(rows, self.meta, self.prov)

    def test_team_mapping_and_only_start_anchor(self):
        events, audit = self.adapt([self.row()])
        f=np.arange(10,101); labels=anchor_labels(events,f,np.ones(len(f)))
        self.assertEqual(labels['team'][10],0)
        self.assertEqual(np.sum(labels['team']>=0),1)
        self.assertTrue((labels['acquisition']==-1).all())
        self.assertIsNone(audit['full_spell_f1'])

    def test_explicit_reception_negative_and_recovery_positive(self):
        rows=[self.row(),self.row(30,8),self.row(40,9,20,start_type='recovery',first_player_possession_in_team_possession='True')]
        events,_=self.adapt(rows[::-1])
        self.assertEqual([e['acquisition_label'] for e in events],[-1,0,1])

    def test_unmatched_and_extrapolated_are_unknown(self):
        for change in [dict(is_player_possession_start_matched='False'),dict(fully_extrapolated='True'),dict(fully_extrapolated='')]:
            events,audit=self.adapt([self.row(**change)])
            self.assertEqual(audit['eligible'],0)
            self.assertEqual(events[0]['contact_label'],-1)

    def test_no_acquisition_across_half_or_interruption(self):
        for second in [self.row(120,9,20,period='2',start_type='recovery',first_player_possession_in_team_possession='True'),
                       self.row(40,9,20,start_type='recovery',first_player_possession_in_team_possession='True',game_interruption_before='throw_in_for')]:
            events,_=self.adapt([self.row(),second]);self.assertEqual(events[1]['acquisition_label'],-1)

    def test_pressure_not_a_control_event_and_no_absence_negatives(self):
        events,audit=self.adapt([{'event_type':'on_ball_engagement'}])
        self.assertEqual(events,[])
        self.assertFalse(audit['missing_event_is_negative'])

    def test_actor_team_mismatch_fails(self):
        with self.assertRaises(ValueError):self.adapt([self.row(team=20)])

    def test_frame_contract_and_zero_duration_one_touch(self):
        events,audit=self.adapt([self.row(frame_end='20',duration='0')])
        self.assertEqual(audit['one_frame'],1)
        for bad in [dict(duration='1'),dict(frame_start='20.5'),dict(period='2'),dict(duration='nan')]:
            with self.assertRaises(ValueError):self.adapt([self.row(**bad)])

    def test_event_derived_fields_do_not_become_features(self):
        a,_=self.adapt([self.row()]);b,_=self.adapt([self.row(overall_pressure_start='very_high_pressure',pass_reception_epv_for='1',x_start='999')])
        self.assertEqual(a,b)
        self.assertNotIn('overall_pressure_start',a[0])

    def test_duplicate_and_mixed_match_fails(self):
        for rows in [[self.row(),self.row()],[self.row(match_id='98')],[self.row(),self.row(event_id='another')]]:
            with self.assertRaises(ValueError):self.adapt(rows)

    def test_unconfirmed_same_team_not_negative(self):
        events,_=self.adapt([self.row(player_targeted_id='9'),self.row(30,8)])
        self.assertEqual(events[1]['acquisition_label'],-1)

    def test_provenance_is_required(self):
        p=copy.deepcopy(self.prov);p['source_kind']='target'
        with self.assertRaises(ValueError):adapt_skillcorner_events([self.row()],self.meta,p)

    def test_half_alignment_fail_closed(self):
        events,_=self.adapt([self.row()])
        with self.assertRaises(ValueError):anchor_labels(events,np.arange(100),np.full(100,2))


if __name__ == '__main__':
    unittest.main()
