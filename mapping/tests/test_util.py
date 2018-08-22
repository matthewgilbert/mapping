import unittest
import os
from mapping import util
from pandas.util.testing import assert_frame_equal, assert_series_equal
import pandas as pd
from pandas import Timestamp as TS
import numpy as np


class TestUtil(unittest.TestCase):

    def setUp(self):
        cdir = os.path.dirname(__file__)
        path = os.path.join(cdir, 'data/')
        files = ["CME-FVU2014.csv", "CME-FVZ2014.csv"]
        self.prices = [os.path.join(path, f) for f in files]

    def tearDown(self):
        pass

    def assert_dict_of_frames(self, dict1, dict2):
        self.assertEquals(dict1.keys(), dict2.keys())
        for key in dict1:
            assert_frame_equal(dict1[key], dict2[key])

    def test_read_price_data(self):
        # using default name_func in read_price_data()
        df = util.read_price_data(self.prices)
        dt1 = TS("2014-09-30")
        dt2 = TS("2014-10-01")
        idx = pd.MultiIndex.from_tuples([(dt1, "CME-FVU2014"),
                                         (dt1, "CME-FVZ2014"),
                                         (dt2, "CME-FVZ2014")],
                                        names=["date", "contract"])
        df_exp = pd.DataFrame([119.27344, 118.35938, 118.35938],
                              index=idx, columns=["Open"])
        assert_frame_equal(df, df_exp)

        def name_func(fstr):
            file_name = os.path.split(fstr)[-1]
            name = file_name.split('-')[1].split('.')[0]
            return name[-4:] + name[:3]

        df = util.read_price_data(self.prices, name_func)
        dt1 = TS("2014-09-30")
        dt2 = TS("2014-10-01")
        idx = pd.MultiIndex.from_tuples([(dt1, "2014FVU"), (dt1, "2014FVZ"),
                                         (dt2, "2014FVZ")],
                                        names=["date", "contract"])
        df_exp = pd.DataFrame([119.27344, 118.35938, 118.35938],
                              index=idx, columns=["Open"])
        assert_frame_equal(df, df_exp)

    def test_calc_rets_one_generic(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5'),
                                         (TS('2015-01-05'), 'CLG5')])
        rets = pd.Series([0.1, 0.05, 0.1, 0.8], index=idx)
        vals = [1, 0.5, 0.5, 1]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLG5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([0.1, 0.075, 0.8],
                                 index=weights.index.levels[0],
                                 columns=['CL1'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_two_generics(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLH5'),
                                         (TS('2015-01-05'), 'CLG5'),
                                         (TS('2015-01-05'), 'CLH5')])
        rets = pd.Series([0.1, 0.15, 0.05, 0.1, 0.8, -0.5, 0.2], index=idx)
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5'),
                                          (TS('2015-01-05'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLH5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1', 'CL2'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([[0.1, 0.15], [0.075, 0.45], [-0.5, 0.2]],
                                 index=weights.index.levels[0],
                                 columns=['CL1', 'CL2'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_two_generics_nans_in_second_generic(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLH5'),
                                         (TS('2015-01-05'), 'CLG5'),
                                         (TS('2015-01-05'), 'CLH5')])
        rets = pd.Series([0.1, np.NaN, 0.05, 0.1, np.NaN, -0.5, 0.2],
                         index=idx)
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5'),
                                          (TS('2015-01-05'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLH5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1', 'CL2'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([[0.1, np.NaN], [0.075, np.NaN], [-0.5, 0.2]],
                                 index=weights.index.levels[0],
                                 columns=['CL1', 'CL2'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_two_generics_non_unique_columns(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLH5'),
                                         (TS('2015-01-05'), 'CLG5'),
                                         (TS('2015-01-05'), 'CLH5')])
        rets = pd.Series([0.1, 0.15, 0.05, 0.1, 0.8, -0.5, 0.2], index=idx)
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5'),
                                          (TS('2015-01-05'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLH5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1', 'CL1'])
        self.assertRaises(ValueError, util.calc_rets, rets, weights)

    def test_calc_rets_two_generics_two_asts(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5'),
                                         (TS('2015-01-04'), 'CLH5'),
                                         (TS('2015-01-05'), 'CLG5'),
                                         (TS('2015-01-05'), 'CLH5')])
        rets1 = pd.Series([0.1, 0.15, 0.05, 0.1, 0.8, -0.5, 0.2], index=idx)
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'COF5'),
                                         (TS('2015-01-03'), 'COG5'),
                                         (TS('2015-01-04'), 'COF5'),
                                         (TS('2015-01-04'), 'COG5'),
                                         (TS('2015-01-04'), 'COH5')])
        rets2 = pd.Series([0.1, 0.15, 0.05, 0.1, 0.4], index=idx)
        rets = {"CL": rets1, "CO": rets2}

        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5'),
                                          (TS('2015-01-05'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLH5')
                                          ])
        weights1 = pd.DataFrame(vals, index=widx, columns=["CL0", "CL1"])
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'COF5'),
                                          (TS('2015-01-03'), 'COG5'),
                                          (TS('2015-01-04'), 'COF5'),
                                          (TS('2015-01-04'), 'COG5'),
                                          (TS('2015-01-04'), 'COH5')
                                          ])
        weights2 = pd.DataFrame(vals, index=widx, columns=["CO0", "CO1"])
        weights = {"CL": weights1, "CO": weights2}
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([[0.1, 0.15, 0.1, 0.15],
                                  [0.075, 0.45, 0.075, 0.25],
                                  [-0.5, 0.2, pd.np.NaN, pd.np.NaN]],
                                 index=weights["CL"].index.levels[0],
                                 columns=['CL0', 'CL1', 'CO0', 'CO1'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_missing_instr_rets_key_error(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5')])
        irets = pd.Series([0.02, 0.01, 0.012], index=idx)
        vals = [1, 1/2, 1/2, 1]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLG5')])
        weights = pd.DataFrame(vals, index=widx, columns=["CL1"])
        self.assertRaises(KeyError, util.calc_rets, irets, weights)

    def test_calc_rets_nan_instr_rets(self):
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLG5'),
                                         (TS('2015-01-05'), 'CLG5')])
        rets = pd.Series([pd.np.NaN, pd.np.NaN, 0.1, 0.8], index=idx)
        vals = [1, 0.5, 0.5, 1]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-05'), 'CLG5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([pd.np.NaN, pd.np.NaN, 0.8],
                                 index=weights.index.levels[0],
                                 columns=['CL1'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_missing_weight(self):
        # see https://github.com/matthewgilbert/mapping/issues/8

        # missing weight for return
        idx = pd.MultiIndex.from_tuples([
            (TS('2015-01-02'), 'CLF5'),
            (TS('2015-01-03'), 'CLF5'),
            (TS('2015-01-04'), 'CLF5')
        ])
        rets = pd.Series([0.02, -0.03, 0.06], index=idx)
        vals = [1, 1]
        widx = pd.MultiIndex.from_tuples([
            (TS('2015-01-02'), 'CLF5'),
            (TS('2015-01-04'), 'CLF5')
        ])
        weights = pd.DataFrame(vals, index=widx, columns=["CL1"])
        self.assertRaises(ValueError, util.calc_rets, rets, weights)

        # extra instrument
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5')])
        weights1 = pd.DataFrame(1, index=idx, columns=["CL1"])
        idx = pd.MultiIndex.from_tuples([
            (TS('2015-01-02'), 'CLF5'),
            (TS('2015-01-02'), 'CLH5'),
            (TS('2015-01-03'), 'CLH5'),  # extra day for no weight instrument
            (TS('2015-01-04'), 'CLF5'),
            (TS('2015-01-04'), 'CLH5')
        ])
        rets = pd.Series([0.02, -0.03, 0.06, 0.05, 0.01], index=idx)
        self.assertRaises(ValueError, util.calc_rets, rets, weights1)

        # leading / trailing returns
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5')])
        weights2 = pd.DataFrame(1, index=idx, columns=["CL1"])
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-01'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-05'), 'CLF5')])
        rets = pd.Series([0.02, -0.03, 0.06, 0.05], index=idx)
        self.assertRaises(ValueError, util.calc_rets, rets, weights2)

    def test_to_notional_empty(self):
        instrs = pd.Series()
        prices = pd.Series()
        multipliers = pd.Series()
        res_exp = pd.Series()
        res = util.to_notional(instrs, prices, multipliers)
        assert_series_equal(res, res_exp)

    def test_to_notional_same_fx(self):
        instrs = pd.Series([-1, 2, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        prices = pd.Series([30.20, 30.5, 10.2], index=['CLZ6', 'COZ6', 'GCZ6'])
        multipliers = pd.Series([1, 1, 1], index=['CLZ6', 'COZ6', 'GCZ6'])

        res_exp = pd.Series([-30.20, 2 * 30.5, 10.2],
                            index=['CLZ6', 'COZ6', 'GCZ6'])

        res = util.to_notional(instrs, prices, multipliers)
        assert_series_equal(res, res_exp)

    def test_to_notional_extra_prices(self):
        instrs = pd.Series([-1, 2, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        multipliers = pd.Series([1, 1, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        prices = pd.Series([30.20, 30.5, 10.2, 13.1], index=['CLZ6', 'COZ6',
                                                             'GCZ6', 'extra'])

        res_exp = pd.Series([-30.20, 2 * 30.5, 10.2],
                            index=['CLZ6', 'COZ6', 'GCZ6'])

        res = util.to_notional(instrs, prices, multipliers)
        assert_series_equal(res, res_exp)

    def test_to_notional_missing_prices(self):
        instrs = pd.Series([-1, 2, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        multipliers = pd.Series([1, 1, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        prices = pd.Series([30.20, 30.5], index=['CLZ6', 'COZ6'])

        res_exp = pd.Series([-30.20, 2 * 30.5, pd.np.NaN],
                            index=['CLZ6', 'COZ6', 'GCZ6'])

        res = util.to_notional(instrs, prices, multipliers)
        assert_series_equal(res, res_exp)

    def test_to_notional_different_fx(self):
        instrs = pd.Series([-1, 2, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        multipliers = pd.Series([1, 1, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        prices = pd.Series([30.20, 30.5, 10.2], index=['CLZ6', 'COZ6', 'GCZ6'])
        instr_fx = pd.Series(['USD', 'CAD', 'AUD'],
                             index=['CLZ6', 'COZ6', 'GCZ6'])
        fx_rates = pd.Series([1.32, 0.8], index=['USDCAD', 'AUDUSD'])

        res_exp = pd.Series([-30.20, 2 * 30.5 / 1.32, 10.2 * 0.8],
                            index=['CLZ6', 'COZ6', 'GCZ6'])

        res = util.to_notional(instrs, prices, multipliers, desired_ccy='USD',
                               instr_fx=instr_fx, fx_rates=fx_rates)
        assert_series_equal(res, res_exp)

    def test_to_notional_duplicates(self):
        instrs = pd.Series([1, 1], index=['A', 'A'])
        prices = pd.Series([200.37], index=['A'])
        mults = pd.Series([100], index=['A'])
        self.assertRaises(ValueError, util.to_notional, instrs, prices, mults)

        instrs = pd.Series([1], index=['A'])
        prices = pd.Series([200.37, 200.37], index=['A', 'A'])
        mults = pd.Series([100], index=['A'])
        self.assertRaises(ValueError, util.to_notional, instrs, prices, mults)

        instrs = pd.Series([1], index=['A'])
        prices = pd.Series([200.37], index=['A'])
        mults = pd.Series([100, 100], index=['A', 'A'])
        self.assertRaises(ValueError, util.to_notional, instrs, prices, mults)

        instrs = pd.Series([1], index=['A'])
        prices = pd.Series([200.37], index=['A'])
        mults = pd.Series([100], index=['A'])
        desired_ccy = "CAD"
        instr_fx = pd.Series(['USD', 'USD'], index=['A', 'A'])
        fx_rate = pd.Series([1.32], index=['USDCAD'])
        self.assertRaises(ValueError, util.to_notional, instrs, prices, mults,
                          desired_ccy, instr_fx, fx_rate)

        instrs = pd.Series([1], index=['A'])
        prices = pd.Series([200.37], index=['A'])
        mults = pd.Series([100], index=['A'])
        desired_ccy = "CAD"
        instr_fx = pd.Series(['USD'], index=['A'])
        fx_rate = pd.Series([1.32, 1.32], index=['USDCAD', 'USDCAD'])
        self.assertRaises(ValueError, util.to_notional, instrs, prices, mults,
                          desired_ccy, instr_fx, fx_rate)

    def test_to_notional_bad_fx(self):
        instrs = pd.Series([1], index=['A'])
        prices = pd.Series([200.37], index=['A'])
        mults = pd.Series([100], index=['A'])
        instr_fx = pd.Series(['JPY'], index=['A'])
        fx_rates = pd.Series([1.32], index=['GBPCAD'])
        self.assertRaises(ValueError, util.to_notional, instrs, prices, mults,
                          desired_ccy='USD', instr_fx=instr_fx,
                          fx_rates=fx_rates)

    def test_to_contracts_rounder(self):
        prices = pd.Series([30.20, 30.5], index=['CLZ6', 'COZ6'])
        multipliers = pd.Series([1, 1], index=['CLZ6', 'COZ6'])
        # 30.19 / 30.20 is slightly less than 1 so will round to 0
        notional = pd.Series([30.19, 2 * 30.5], index=['CLZ6', 'COZ6'])
        res = util.to_contracts(notional, prices, multipliers,
                                rounder=pd.np.floor)
        res_exp = pd.Series([0, 2], index=['CLZ6', 'COZ6'])
        assert_series_equal(res, res_exp)

    def test_to_contract_different_fx_with_multiplier(self):
        notionals = pd.Series([-30.20, 2 * 30.5 / 1.32 * 10, 10.2 * 0.8 * 100],
                              index=['CLZ6', 'COZ6', 'GCZ6'])
        prices = pd.Series([30.20, 30.5, 10.2], index=['CLZ6', 'COZ6', 'GCZ6'])
        instr_fx = pd.Series(['USD', 'CAD', 'AUD'],
                             index=['CLZ6', 'COZ6', 'GCZ6'])
        fx_rates = pd.Series([1.32, 0.8], index=['USDCAD', 'AUDUSD'])
        multipliers = pd.Series([1, 10, 100], index=['CLZ6', 'COZ6', 'GCZ6'])

        res_exp = pd.Series([-1, 2, 1], index=['CLZ6', 'COZ6', 'GCZ6'])

        res = util.to_contracts(notionals, prices, desired_ccy='USD',
                                instr_fx=instr_fx, fx_rates=fx_rates,
                                multipliers=multipliers)
        assert_series_equal(res, res_exp)

    def test_to_contract_different_fx_with_multiplier_rounding(self):
        # won't work out to integer number of contracts so this tests rounding
        notionals = pd.Series([-30.21, 2 * 30.5 / 1.32 * 10, 10.2 * 0.8 * 100],
                              index=['CLZ6', 'COZ6', 'GCZ6'])
        prices = pd.Series([30.20, 30.5, 10.2], index=['CLZ6', 'COZ6', 'GCZ6'])
        instr_fx = pd.Series(['USD', 'CAD', 'AUD'],
                             index=['CLZ6', 'COZ6', 'GCZ6'])
        fx_rates = pd.Series([1.32, 0.8], index=['USDCAD', 'AUDUSD'])
        multipliers = pd.Series([1, 10, 100], index=['CLZ6', 'COZ6', 'GCZ6'])

        res_exp = pd.Series([-1, 2, 1], index=['CLZ6', 'COZ6', 'GCZ6'])
        res = util.to_contracts(notionals, prices, desired_ccy='USD',
                                instr_fx=instr_fx, fx_rates=fx_rates,
                                multipliers=multipliers)
        assert_series_equal(res, res_exp)

    def test_trade_with_zero_amount(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=["CLX16", "CLZ16", "CLF17"],
                           columns=[0, 1])
        desired_holdings = pd.Series([200000, 0], index=[0, 1])
        current_contracts = pd.Series([0, 1, 0],
                                      index=['CLX16', 'CLZ16', 'CLF17'])
        prices = pd.Series([50.32, 50.41, 50.48],
                           index=['CLX16', 'CLZ16', 'CLF17'])
        multiplier = pd.Series([100, 100, 100],
                               index=['CLX16', 'CLZ16', 'CLF17'])
        trades = util.calc_trades(current_contracts, desired_holdings, wts,
                                  prices, multipliers=multiplier)
        # 200000 * 0.5 / (50.32*100) - 0,
        # 200000 * 0.5 / (50.41*100) + 0 * 0.5 / (50.41*100) - 1,
        # 0 * 0.5 / (50.48*100) - 0,

        exp_trades = pd.Series([20, 19], index=['CLX16', 'CLZ16'])
        assert_series_equal(trades, exp_trades)

    def test_trade_all_zero_amount_return_empty(self):
        wts = pd.DataFrame([1], index=["CLX16"], columns=[0])
        desired_holdings = pd.Series([13], index=[0])
        current_contracts = 0
        prices = pd.Series([50.32], index=['CLX16'])
        multiplier = pd.Series([100], index=['CLX16'])
        trades = util.calc_trades(current_contracts, desired_holdings, wts,
                                  prices, multipliers=multiplier)

        exp_trades = pd.Series(dtype="int64")
        assert_series_equal(trades, exp_trades)

    def test_trade_one_asset(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=["CLX16", "CLZ16", "CLF17"],
                           columns=[0, 1])
        desired_holdings = pd.Series([200000, -50000], index=[0, 1])
        current_contracts = pd.Series([0, 1, 0],
                                      index=['CLX16', 'CLZ16', 'CLF17'])
        prices = pd.Series([50.32, 50.41, 50.48],
                           index=['CLX16', 'CLZ16', 'CLF17'])
        multiplier = pd.Series([100, 100, 100],
                               index=['CLX16', 'CLZ16', 'CLF17'])
        trades = util.calc_trades(current_contracts, desired_holdings, wts,
                                  prices, multipliers=multiplier)
        # 200000 * 0.5 / (50.32*100) - 0,
        # 200000 * 0.5 / (50.41*100) - 50000 * 0.5 / (50.41*100) - 1,
        # -50000 * 0.5 / (50.48*100) - 0,

        exp_trades = pd.Series([20, 14, -5], index=['CLX16', 'CLZ16', 'CLF17'])
        exp_trades = exp_trades.sort_index()
        assert_series_equal(trades, exp_trades)

    def test_trade_multi_asset(self):
        wts1 = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                            index=["CLX16", "CLZ16", "CLF17"],
                            columns=["CL0", "CL1"])
        wts2 = pd.DataFrame([1], index=["COX16"], columns=["CO0"])
        wts = {"CL": wts1, "CO": wts2}
        desired_holdings = pd.Series([200000, -50000, 100000],
                                     index=["CL0", "CL1", "CO0"])
        current_contracts = pd.Series([0, 1, 0, 5],
                                      index=['CLX16', 'CLZ16', 'CLF17',
                                             'COX16'])
        prices = pd.Series([50.32, 50.41, 50.48, 49.50],
                           index=['CLX16', 'CLZ16', 'CLF17', 'COX16'])
        multiplier = pd.Series([100, 100, 100, 100],
                               index=['CLX16', 'CLZ16', 'CLF17', 'COX16'])
        trades = util.calc_trades(current_contracts, desired_holdings, wts,
                                  prices, multipliers=multiplier)
        # 200000 * 0.5 / (50.32*100) - 0,
        # 200000 * 0.5 / (50.41*100) - 50000 * 0.5 / (50.41*100) - 1,
        # -50000 * 0.5 / (50.48*100) - 0,
        # 100000 * 1 / (49.50*100) - 5,

        exp_trades = pd.Series([20, 14, -5, 15], index=['CLX16', 'CLZ16',
                                                        'CLF17', 'COX16'])
        exp_trades = exp_trades.sort_index()
        assert_series_equal(trades, exp_trades)

    def test_trade_extra_desired_holdings_without_weights(self):
        wts = pd.DataFrame([0], index=["CLX16"], columns=["CL0"])
        desired_holdings = pd.Series([200000, 10000], index=["CL0", "CL1"])
        current_contracts = pd.Series([0], index=['CLX16'])
        prices = pd.Series([50.32], index=['CLX16'])
        multipliers = pd.Series([1], index=['CLX16'])
        self.assertRaises(ValueError, util.calc_trades, current_contracts,
                          desired_holdings, wts, prices, multipliers)

    def test_trade_extra_desired_holdings_without_current_contracts(self):
        # this should treat the missing holdings as 0, since this would often
        # happen when adding new positions without any current holdings
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=["CLX16", "CLZ16", "CLF17"],
                           columns=[0, 1])
        desired_holdings = pd.Series([200000, -50000], index=[0, 1])
        current_contracts = pd.Series([0, 1],
                                      index=['CLX16', 'CLZ16'])
        prices = pd.Series([50.32, 50.41, 50.48],
                           index=['CLX16', 'CLZ16', 'CLF17'])
        multiplier = pd.Series([100, 100, 100],
                               index=['CLX16', 'CLZ16', 'CLF17'])
        trades = util.calc_trades(current_contracts, desired_holdings, wts,
                                  prices, multipliers=multiplier)
        # 200000 * 0.5 / (50.32*100) - 0,
        # 200000 * 0.5 / (50.41*100) - 50000 * 0.5 / (50.41*100) - 1,
        # -50000 * 0.5 / (50.48*100) - 0,

        exp_trades = pd.Series([20, 14, -5], index=['CLX16', 'CLZ16', 'CLF17'])
        exp_trades = exp_trades.sort_index()
        # non existent contract holdings result in fill value being a float,
        # which casts to float64
        assert_series_equal(trades, exp_trades, check_dtype=False)

    def test_trade_extra_weights(self):
        # extra weights should be ignored
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=["CLX16", "CLZ16", "CLF17"],
                           columns=[0, 1])
        desired_holdings = pd.Series([200000], index=[0])
        current_contracts = pd.Series([0, 2], index=['CLX16', 'CLZ16'])
        prices = pd.Series([50.32, 50.41], index=['CLX16', 'CLZ16'])
        multiplier = pd.Series([100, 100], index=['CLX16', 'CLZ16'])
        trades = util.calc_trades(current_contracts, desired_holdings, wts,
                                  prices, multipliers=multiplier)
        # 200000 * 0.5 / (50.32*100) - 0,
        # 200000 * 0.5 / (50.41*100) - 2,

        exp_trades = pd.Series([20, 18], index=['CLX16', 'CLZ16'])
        assert_series_equal(trades, exp_trades)

    def test_get_multiplier_dataframe_weights(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=["CLX16", "CLZ16", "CLF17"],
                           columns=[0, 1])
        ast_mult = pd.Series([1000], index=["CL"])

        imults = util.get_multiplier(wts, ast_mult)
        imults_exp = pd.Series([1000, 1000, 1000],
                               index=["CLF17", "CLX16", "CLZ16"])
        assert_series_equal(imults, imults_exp)

    def test_get_multiplier_dict_weights(self):
        wts1 = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                            index=["CLX16", "CLZ16", "CLF17"],
                            columns=[0, 1])
        wts2 = pd.DataFrame([0.5, 0.5], index=["COX16", "COZ16"], columns=[0])
        wts = {"CL": wts1, "CO": wts2}
        ast_mult = pd.Series([1000, 1000], index=["CL", "CO"])

        imults = util.get_multiplier(wts, ast_mult)
        imults_exp = pd.Series([1000, 1000, 1000, 1000, 1000],
                               index=["CLF17", "CLX16", "CLZ16", "COX16",
                                      "COZ16"])
        assert_series_equal(imults, imults_exp)

    def test_get_multiplier_dataframe_weights_multiplier_asts_error(self):
        wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
                           index=["CLX16", "CLZ16", "CLF17"],
                           columns=[0, 1])
        ast_mult = pd.Series([1000, 1000], index=["CL", "CO"])
        self.assertRaises(ValueError, util.get_multiplier, wts, ast_mult)

    def test_weighted_expiration_two_generics(self):
        vals = [[1, 0, 1/2, 1/2, 0, 1, 0], [0, 1, 0, 1/2, 1/2, 0, 1]]
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF15'),
                                         (TS('2015-01-03'), 'CLG15'),
                                         (TS('2015-01-04'), 'CLF15'),
                                         (TS('2015-01-04'), 'CLG15'),
                                         (TS('2015-01-04'), 'CLH15'),
                                         (TS('2015-01-05'), 'CLG15'),
                                         (TS('2015-01-05'), 'CLH15')])
        weights = pd.DataFrame({"CL1": vals[0], "CL2": vals[1]}, index=idx)
        contract_dates = pd.Series([TS('2015-01-20'),
                                    TS('2015-02-21'),
                                    TS('2015-03-20')],
                                   index=['CLF15', 'CLG15', 'CLH15'])
        wexp = util.weighted_expiration(weights, contract_dates)
        exp_wexp = pd.DataFrame([[17.0, 49.0], [32.0, 61.5], [47.0, 74.0]],
                                index=[TS('2015-01-03'),
                                       TS('2015-01-04'),
                                       TS('2015-01-05')],
                                columns=["CL1", "CL2"])
        assert_frame_equal(wexp, exp_wexp)

    def test_flatten(self):
        vals = [[1, 0], [0, 1], [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5')])
        weights = pd.DataFrame(vals, index=widx, columns=["CL1", "CL2"])
        flat_wts = util.flatten(weights)

        flat_wts_exp = pd.DataFrame(
            {"date": [TS('2015-01-03')] * 4 + [TS('2015-01-04')] * 4,
             "contract": ['CLF5'] * 2 + ['CLG5'] * 4 + ['CLH5'] * 2,
             "generic": ["CL1", "CL2"] * 4,
             "weight": [1, 0, 0, 1, 1, 0, 0, 1]}
        ).loc[:, ["date", "contract", "generic", "weight"]]
        assert_frame_equal(flat_wts, flat_wts_exp)

    def test_flatten_dict(self):
        vals = [[1, 0], [0, 1], [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5')])
        weights1 = pd.DataFrame(vals, index=widx, columns=["CL1", "CL2"])
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'COF5')])
        weights2 = pd.DataFrame(1, index=widx, columns=["CO1"])
        weights = {"CL": weights1, "CO": weights2}
        flat_wts = util.flatten(weights)
        flat_wts_exp = pd.DataFrame(
            {"date": ([TS('2015-01-03')] * 4 + [TS('2015-01-04')] * 4
                      + [TS('2015-01-03')]),
             "contract": (['CLF5'] * 2 + ['CLG5'] * 4 + ['CLH5'] * 2
                          + ["COF5"]),
             "generic": ["CL1", "CL2"] * 4 + ["CO1"],
             "weight": [1, 0, 0, 1, 1, 0, 0, 1, 1],
             "key": ["CL"] * 8 + ["CO"]}
        ).loc[:, ["date", "contract", "generic", "weight", "key"]]
        assert_frame_equal(flat_wts, flat_wts_exp)

    def test_flatten_bad_input(self):
        dummy = 0
        self.assertRaises(ValueError, util.flatten, dummy)

    def test_unflatten(self):
        flat_wts = pd.DataFrame(
            {"date": [TS('2015-01-03')] * 4 + [TS('2015-01-04')] * 4,
             "contract": ['CLF5'] * 2 + ['CLG5'] * 4 + ['CLH5'] * 2,
             "generic": ["CL1", "CL2"] * 4,
             "weight": [1, 0, 0, 1, 1, 0, 0, 1]}
        ).loc[:, ["date", "contract", "generic", "weight"]]
        wts = util.unflatten(flat_wts)

        vals = [[1, 0], [0, 1], [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5')],
                                         names=("date", "contract"))
        cols = pd.Index(["CL1", "CL2"], name="generic")
        wts_exp = pd.DataFrame(vals, index=widx, columns=cols)
        assert_frame_equal(wts, wts_exp)

    def test_unflatten_dict(self):
        flat_wts = pd.DataFrame(
            {"date": ([TS('2015-01-03')] * 4 + [TS('2015-01-04')] * 4
                      + [TS('2015-01-03')]),
             "contract": (['CLF5'] * 2 + ['CLG5'] * 4 + ['CLH5'] * 2
                          + ["COF5"]),
             "generic": ["CL1", "CL2"] * 4 + ["CO1"],
             "weight": [1, 0, 0, 1, 1, 0, 0, 1, 1],
             "key": ["CL"] * 8 + ["CO"]}
        ).loc[:, ["date", "contract", "generic", "weight", "key"]]
        wts = util.unflatten(flat_wts)

        vals = [[1, 0], [0, 1], [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLG5'),
                                          (TS('2015-01-04'), 'CLH5')],
                                         names=("date", "contract"))
        cols = pd.Index(["CL1", "CL2"], name="generic")
        weights1 = pd.DataFrame(vals, index=widx, columns=cols)
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'COF5')],
                                         names=("date", "contract"))
        cols = pd.Index(["CO1"], name="generic")
        weights2 = pd.DataFrame(1, index=widx, columns=cols)
        wts_exp = {"CL": weights1, "CO": weights2}

        self.assert_dict_of_frames(wts, wts_exp)

    def test_reindex(self):
        # related to https://github.com/matthewgilbert/mapping/issues/11
        # no op
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLH5'),
                                         (TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLH5')])
        prices = pd.Series([103, 101, 102, 100], index=idx)
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLH5')])
        new_prices = util.reindex(prices, widx, limit=0)
        exp_prices = prices
        assert_series_equal(exp_prices, new_prices)

        # missing front prices error
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5')])
        prices = pd.Series([100], index=idx)
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5')])
        self.assertRaises(ValueError, util.reindex, prices, widx, 0)

        # NaN returns introduced and filled
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-01'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLH5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLH5')])
        prices = pd.Series([100, 101, 102, 103, 104], index=idx)
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-02'), 'CLF5'),
                                          (TS('2015-01-02'), 'CLH5'),
                                          (TS('2015-01-04'), 'CLF5'),
                                          (TS('2015-01-04'), 'CLH5'),
                                          (TS('2015-01-05'), 'CLF5'),
                                          (TS('2015-01-05'), 'CLH5'),
                                          (TS('2015-01-06'), 'CLF5'),
                                          (TS('2015-01-06'), 'CLH5')])
        new_prices = util.reindex(prices, widx, limit=1)

        idx = pd.MultiIndex.from_tuples([(TS('2015-01-01'), 'CLF5'),
                                         (TS('2015-01-01'), 'CLH5'),
                                         (TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLH5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLH5'),
                                         (TS('2015-01-05'), 'CLF5'),
                                         (TS('2015-01-05'), 'CLH5'),
                                         (TS('2015-01-06'), 'CLF5'),
                                         (TS('2015-01-06'), 'CLH5')
                                         ])
        exp_prices = pd.Series([100, np.NaN, 101, 102, 103, 104, 103,
                                104, np.NaN, np.NaN], index=idx)
        assert_series_equal(exp_prices, new_prices)

        # standard subset
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-01'), 'CLF5'),
                                         (TS('2015-01-01'), 'CHF5'),
                                         (TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLH5'),
                                         (TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLH5'),
                                         (TS('2015-01-04'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLH5')])
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107], index=idx)
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-02'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLH5')])
        new_prices = util.reindex(prices, widx, limit=0)

        idx = pd.MultiIndex.from_tuples([(TS('2015-01-01'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLF5'),
                                         (TS('2015-01-02'), 'CLH5'),
                                         (TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-03'), 'CLH5')])
        exp_prices = pd.Series([100, 102, 103, 104, 105], index=idx)
        assert_series_equal(exp_prices, new_prices)

        # check unique index to avoid duplicates from pd.Series.reindex
        idx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                         (TS('2015-01-04'), 'CLF5')])
        prices = pd.Series([100.10, 101.13], index=idx)
        widx = pd.MultiIndex.from_tuples([(TS('2015-01-03'), 'CLF5'),
                                          (TS('2015-01-03'), 'CLF5')])
        self.assertRaises(ValueError, util.reindex, prices, widx, limit=0)
