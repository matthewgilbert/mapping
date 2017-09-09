import unittest
from mapping import mappings
from pandas.util.testing import assert_frame_equal, assert_series_equal
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay


class TestMappings(unittest.TestCase):

    def setUp(self):
        self.dates = pd.Series([pd.Timestamp('2016-10-20'),
                                pd.Timestamp('2016-11-21'),
                                pd.Timestamp('2016-12-20')],
                               index=['CLX16', 'CLZ16', 'CLF17'])
        self.short_dates = pd.Series([pd.Timestamp('2016-10-10'),
                                      pd.Timestamp('2016-10-13'),
                                      pd.Timestamp('2016-10-17'),
                                      pd.Timestamp('2016-10-20')],
                                     index=['A', 'B', 'C', 'D'])

    def tearDown(self):
        pass

    def test_not_in_roll_one_generic_static_roller(self):
        dt = self.dates.iloc[0]
        contract_dates = self.dates.iloc[0:2]
        sd, ed = (dt + BDay(-8), dt + BDay(-7))
        timestamps = pd.date_range(sd, ed, freq='b')
        cols = pd.MultiIndex.from_product([[0], ['front', 'back']])
        idx = [-2, -1, 0]
        trans = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                             index=idx, columns=cols)

        midx = pd.MultiIndex.from_product([timestamps, ['CLX16']])
        midx.names = ['date', 'contract']
        cols = pd.Index([0], name='generic')
        wts_exp = pd.DataFrame([1.0, 1.0], index=midx, columns=cols)

        # with DatetimeIndex
        wts = mappings.roller(timestamps, contract_dates,
                              mappings.static_transition, transition=trans)
        assert_frame_equal(wts, wts_exp)

        # with tuple
        wts = mappings.roller(tuple(timestamps), contract_dates,
                              mappings.static_transition, transition=trans)
        assert_frame_equal(wts, wts_exp)

    def test_not_in_roll_one_generic_static_transition(self):
        contract_dates = self.dates.iloc[0:2]
        ts = self.dates.iloc[0] + BDay(-8)
        cols = pd.MultiIndex.from_product([[0], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                  index=idx, columns=cols)

        wts = mappings.static_transition(ts, contract_dates, transition)
        wts_exp = [(0, 'CLX16', 1.0, ts)]
        self.assertEqual(wts, wts_exp)

    def test_non_numeric_column_static_transition(self):
        contract_dates = self.dates.iloc[0:2]
        ts = self.dates.iloc[0] + BDay(-8)
        cols = pd.MultiIndex.from_product([["CL1"], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                  index=idx, columns=cols)

        wts = mappings.static_transition(ts, contract_dates, transition)
        wts_exp = [("CL1", 'CLX16', 1.0, ts)]
        self.assertEqual(wts, wts_exp)

    def test_finished_roll_pre_expiry_static_transition(self):
        contract_dates = self.dates.iloc[0:2]
        ts = self.dates.iloc[0] + BDay(-2)
        cols = pd.MultiIndex.from_product([[0], ['front', 'back']])
        idx = [-9, -8]
        transition = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                                  index=idx, columns=cols)
        wts = mappings.static_transition(ts, contract_dates, transition)
        wts_exp = [(0, 'CLZ16', 1.0, ts)]
        self.assertEqual(wts, wts_exp)

    def test_not_in_roll_one_generic_filtering_front_contracts_static_transition(self):  # NOQA
        contract_dates = self.dates.iloc[0:2]
        ts = self.dates.iloc[1] + BDay(-8)
        cols = pd.MultiIndex.from_product([[0], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                  index=idx, columns=cols)

        wts = mappings.static_transition(ts, contract_dates, transition)
        wts_exp = [(0, 'CLZ16', 1.0, ts)]
        self.assertEqual(wts, wts_exp)

    def test_roll_with_holiday(self):
        contract_dates = self.dates.iloc[-2:]
        ts = pd.Timestamp("2016-11-17")
        cols = pd.MultiIndex.from_product([[0], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                  index=idx, columns=cols)

        holidays = [np.datetime64("2016-11-18")]
        # the holiday moves the roll schedule up one day, since Friday is
        # excluded as a day
        wts = mappings.static_transition(ts, contract_dates, transition,
                                         holidays)
        wts_exp = [(0, 'CLZ16', 0.5, ts), (0, 'CLF17', 0.5, ts)]
        self.assertEqual(wts, wts_exp)

    def test_aggregate_weights(self):
        ts = pd.Timestamp("2015-01-01")
        wts_list = [(0, 'CLX16', 1.0, ts), (1, 'CLZ16', 1.0, ts)]
        wts = mappings.aggregate_weights(wts_list)
        idx = pd.MultiIndex.from_product([[ts], ["CLX16", "CLZ16"]],
                                         names=["date", "contract"])
        cols = pd.Index([0, 1], name="generic")
        wts_exp = pd.DataFrame([[1.0, 0], [0, 1.0]], index=idx, columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_aggregate_weights_drop_date(self):
        ts = pd.Timestamp("2015-01-01")
        wts_list = [(0, 'CLX16', 1.0, ts), (1, 'CLZ16', 1.0, ts)]
        wts = mappings.aggregate_weights(wts_list, drop_date=True)
        idx = pd.Index(["CLX16", "CLZ16"], name="contract")
        cols = pd.Index([0, 1], name="generic")
        wts_exp = pd.DataFrame([[1.0, 0], [0, 1.0]], index=idx, columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_not_in_roll_one_generic_zero_weight_back_contract_no_contract_static_transition(self):  # NOQA
        contract_dates = self.dates.iloc[0:1]
        ts = self.dates.iloc[0] + BDay(-8)
        cols = pd.MultiIndex.from_product([[0], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                  index=idx, columns=cols)

        wts = mappings.static_transition(ts, contract_dates, transition)
        wts_exp = [(0, 'CLX16', 1.0, ts)]
        self.assertEqual(wts, wts_exp)

    def test_static_bad_transition(self):
        contract_dates = self.dates.iloc[[0]]
        ts = self.dates.iloc[0] + BDay(-8)
        cols = pd.MultiIndex.from_product([[0], ['not_front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                  index=idx, columns=cols)

        def bad_transition():
            mappings.static_transition(ts, contract_dates, transition)

        self.assertRaises(ValueError, bad_transition)

    def test_no_roll_date_two_generics_static_transition(self):
        dt = self.dates.iloc[0]
        contract_dates = self.dates
        ts = dt + BDay(-8)
        cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5],
                                   [0.0, 1.0, 0.0, 1.0]],
                                  index=idx, columns=cols)

        wts = mappings.static_transition(ts, contract_dates, transition)
        wts_exp = [(0, 'CLX16', 1.0, ts), (1, 'CLZ16', 1.0, ts)]
        self.assertEqual(wts, wts_exp)

    def test_not_in_roll_two_generics_static_roller(self):
        dt = self.dates.iloc[0]
        contract_dates = self.dates.iloc[0:3]
        sd, ed = (dt + BDay(-8), dt + BDay(-7))
        timestamps = pd.date_range(sd, ed, freq='b')
        cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5],
                                   [0.0, 1.0, 0.0, 1.0]],
                                  index=idx, columns=cols)

        wts = mappings.roller(timestamps, contract_dates,
                              mappings.static_transition,
                              transition=transition)

        midx = pd.MultiIndex.from_product([timestamps, ['CLX16', 'CLZ16']])
        midx.names = ['date', 'contract']
        cols = pd.Index([0, 1], name='generic')
        wts_exp = pd.DataFrame([[1.0, 0.0], [0.0, 1.0],
                                [1.0, 0.0], [0.0, 1.0]], index=midx,
                               columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_during_roll_two_generics_one_day_static_transition(self):
        contract_dates = self.dates
        ts = self.dates.iloc[0] + BDay(-1)
        cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
        idx = [-2, -1, 0]
        transition = pd.DataFrame([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5],
                                   [0.0, 1.0, 0.0, 1.0]],
                                  index=idx, columns=cols)
        wts = mappings.static_transition(ts, contract_dates, transition)

        wts_exp = [(0, 'CLX16', 0.5, ts), (0, 'CLZ16', 0.5, ts),
                   (1, 'CLZ16', 0.5, ts), (1, 'CLF17', 0.5, ts)]

        self.assertEqual(wts, wts_exp)

    def test_during_roll_two_generics_one_day_static_roller(self):
        dt = self.dates.iloc[0]
        contract_dates = self.dates
        timestamps = pd.DatetimeIndex([dt + BDay(-1)])
        cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
        idx = [-2, -1, 0]
        trans = pd.DataFrame([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5],
                              [0.0, 1.0, 0.0, 1.0]],
                             index=idx, columns=cols)
        wts = mappings.roller(timestamps, contract_dates,
                              mappings.static_transition, transition=trans)

        midx = pd.MultiIndex.from_product([timestamps,
                                           ['CLF17', 'CLX16', 'CLZ16']])
        midx.names = ['date', 'contract']
        cols = pd.Index([0, 1], name='generic')
        wts_exp = pd.DataFrame([[0, 0.5], [0.5, 0], [0.5, 0.5]],
                               index=midx, columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_whole_roll_roll_two_generics_static_roller(self):
        dt = self.dates.iloc[0]
        contract_dates = self.dates
        timestamps = pd.DatetimeIndex([dt + BDay(-2), dt + BDay(-1), dt])
        cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
        idx = [-2, -1, 0]
        trans = pd.DataFrame([[1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5],
                              [0, 1, 0, 1]],
                             index=idx, columns=cols)
        wts = mappings.roller(timestamps, contract_dates,
                              mappings.static_transition, transition=trans)

        midx = pd.MultiIndex.from_tuples([(timestamps[0], 'CLX16'),
                                          (timestamps[0], 'CLZ16'),
                                          (timestamps[1], 'CLF17'),
                                          (timestamps[1], 'CLX16'),
                                          (timestamps[1], 'CLZ16'),
                                          (timestamps[2], 'CLF17'),
                                          (timestamps[2], 'CLZ16')])
        midx.names = ['date', 'contract']
        cols = pd.Index([0, 1], name='generic')
        wts_exp = pd.DataFrame([[1, 0], [0, 1], [0, 0.5], [0.5, 0], [0.5, 0.5],
                                [0, 1], [1, 0]],
                               index=midx, columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_roll_to_roll_two_generics(self):
        contract_dates = self.short_dates
        timestamps = pd.date_range(contract_dates.iloc[0] + BDay(-2),
                                   contract_dates.iloc[1], freq='b')
        cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
        idx = [-2, -1, 0]
        trans = pd.DataFrame([[1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5],
                              [0, 1, 0, 1]], index=idx, columns=cols)

        wts = mappings.roller(timestamps, contract_dates,
                              mappings.static_transition, transition=trans)
        midx = pd.MultiIndex.from_tuples([(timestamps[0], 'A'),
                                          (timestamps[0], 'B'),
                                          (timestamps[1], 'A'),
                                          (timestamps[1], 'B'),
                                          (timestamps[1], 'C'),
                                          (timestamps[2], 'B'),
                                          (timestamps[2], 'C'),
                                          (timestamps[3], 'B'),
                                          (timestamps[3], 'C'),
                                          (timestamps[4], 'B'),
                                          (timestamps[4], 'C'),
                                          (timestamps[4], 'D'),
                                          (timestamps[5], 'C'),
                                          (timestamps[5], 'D')])
        midx.names = ['date', 'contract']
        cols = pd.Index([0, 1], name='generic')
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1],
                [1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        wts_exp = pd.DataFrame(vals, index=midx, columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_to_generics_two_generics_exact_soln(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=['CLX16', 'CLZ16', 'CLF17'],
                           columns=[0, 1])
        instrs = pd.Series([10, 20, 10], index=["CLX16", "CLZ16", "CLF17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([20.0, 20.0], index=[0, 1])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_two_generics_exact_soln_negative(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=['CLX16', 'CLZ16', 'CLF17'],
                           columns=[0, 1])
        instrs = pd.Series([10, 0, -10], index=["CLX16", "CLZ16", "CLF17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([20.0, -20.0], index=[0, 1])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_two_generics_minimize_error_non_integer_soln(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=['CLX16', 'CLZ16', 'CLF17'],
                           columns=[0, 1])
        instrs = pd.Series([10, 20, 11], index=["CLX16", "CLZ16", "CLF17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([19.5, 21.5], index=[0, 1])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_two_generics_minimize_error_integer_soln(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=['CLX16', 'CLZ16', 'CLF17'],
                           columns=[0, 1])
        instrs = pd.Series([10, 25, 11], index=["CLX16", "CLZ16", "CLF17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([22.0, 24.0], index=[0, 1])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_three_generics_exact_soln(self):
        wts = pd.DataFrame([[0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5],
                            [0, 0, 0.5]],
                           index=['CLX16', 'CLZ16', 'CLF17', 'CLG17'],
                           columns=[0, 1, 2])
        instrs = pd.Series([10, 20, 20, 10],
                           index=["CLX16", "CLZ16", "CLF17", "CLG17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([20.0, 20.0, 20.0], index=[0, 1, 2])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_three_generics_non_exact_soln(self):
        wts = pd.DataFrame([[0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5],
                            [0, 0, 0.5]],
                           index=['CLX16', 'CLZ16', 'CLF17', 'CLG17'],
                           columns=[0, 1, 2])
        instrs = pd.Series([10, 21, 20, 13],
                           index=["CLX16", "CLZ16", "CLF17", "CLG17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([22.0, 18.0, 24.0], index=[0, 1, 2])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_two_generics_multi_asset(self):
        wts1 = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                            index=['CLX16', 'CLZ16', 'CLF17'],
                            columns=["CL0", "CL1"])
        wts2 = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                            index=['COX16', 'COZ16', 'COF17'],
                            columns=["CO0", "CO1"])
        wts = {"CL": wts1, "CO": wts2}
        instrs = pd.Series([10, 20, 10, 10, 20, 10],
                           index=["CLX16", "CLZ16", "CLF17",
                                  "COX16", "COZ16", "COF17"])
        generics = mappings.to_generics(instrs, wts)
        exp_generics = pd.Series([20.0, 20.0, 20.0, 20.0],
                                 index=["CL0", "CL1", "CO0", "CO1"])
        assert_series_equal(generics, exp_generics)

    def test_to_generics_two_generics_key_error(self):
        wts1 = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                            index=['CLX16', 'CLZ16', 'CLF17'],
                            columns=[0, 1])
        # COZ16 is mistyped as CLO16 resulting in no weights for the instrument
        # COZ16
        wts2 = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                            index=['COX16', 'CLO16', 'COF17'],
                            columns=[0, 1])
        wts = {"CL": wts1, "CO": wts2}
        instrs = pd.Series([10, 20, 10, 10, 20, 10],
                           index=["CLX16", "CLZ16", "CLF17",
                                  "COX16", "COZ16", "COF17"])

        def map_gen():
            return mappings.to_generics(instrs, wts)

        self.assertRaises(KeyError, map_gen)