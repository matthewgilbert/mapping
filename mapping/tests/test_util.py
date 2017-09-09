import unittest
import os
from mapping import util
from pandas.util.testing import assert_frame_equal, assert_series_equal
import pandas as pd


class TestUtil(unittest.TestCase):

    def setUp(self):
        cdir = os.path.dirname(__file__)
        path = os.path.join(cdir, 'data/')
        files = ["CME-FVU2014.csv", "CME-FVZ2014.csv"]
        self.prices = [os.path.join(path, f) for f in files]

    def tearDown(self):
        pass

    def test_read_price_data(self):

        def name_func(fstr):
            name = fstr.split('-')[1].split('.')[0]
            return name[-4:] + name[:3]

        df = util.read_price_data(self.prices, name_func)
        dt1 = pd.Timestamp("2014-09-30")
        dt2 = pd.Timestamp("2014-10-01")
        idx = pd.MultiIndex.from_tuples([(dt1, "2014FVU"), (dt1, "2014FVZ"),
                                         (dt2, "2014FVZ")])
        df_exp = pd.DataFrame([119.27344, 118.35938, 118.35938],
                              index=idx, columns=["Open"])
        assert_frame_equal(df, df_exp)

    def test_calc_rets_one_generic(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLG5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5')])
        rets = pd.Series([0.1, 0.05, 0.1, 0.8], index=idx)
        vals = [1, 0.5, 0.5, 1]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([0.1, 0.075, 0.8],
                                 index=weights.index.levels[0],
                                 columns=['CL1'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_two_generics(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-03'), 'CLG5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLG5'),
                                         (pd.Timestamp('2015-01-04'), 'CLH5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5'),
                                         (pd.Timestamp('2015-01-05'), 'CLH5')])
        rets = pd.Series([0.1, 0.15, 0.05, 0.1, 0.8, -0.5, 0.2], index=idx)
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-03'), 'CLG5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-04'), 'CLH5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLH5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1', 'CL2'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([[0.1, 0.15], [0.075, 0.45], [-0.5, 0.2]],
                                 index=weights.index.levels[0],
                                 columns=['CL1', 'CL2'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_two_generics_non_unique_columns(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-03'), 'CLG5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLG5'),
                                         (pd.Timestamp('2015-01-04'), 'CLH5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5'),
                                         (pd.Timestamp('2015-01-05'), 'CLH5')])
        rets = pd.Series([0.1, 0.15, 0.05, 0.1, 0.8, -0.5, 0.2], index=idx)
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-03'), 'CLG5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-04'), 'CLH5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLH5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1', 'CL1'])

        def non_unique():
            return util.calc_rets(rets, weights)

        self.assertRaises(ValueError, non_unique)

    def test_calc_rets_two_generics_two_asts(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-03'), 'CLG5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLG5'),
                                         (pd.Timestamp('2015-01-04'), 'CLH5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5'),
                                         (pd.Timestamp('2015-01-05'), 'CLH5')])
        rets1 = pd.Series([0.1, 0.15, 0.05, 0.1, 0.8, -0.5, 0.2], index=idx)
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'COF5'),
                                         (pd.Timestamp('2015-01-03'), 'COG5'),
                                         (pd.Timestamp('2015-01-04'), 'COF5'),
                                         (pd.Timestamp('2015-01-04'), 'COG5'),
                                         (pd.Timestamp('2015-01-04'), 'COH5')])
        rets2 = pd.Series([0.1, 0.15, 0.05, 0.1, 0.4], index=idx)
        rets = {"CL": rets1, "CO": rets2}

        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5],
                [1, 0], [0, 1]]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-03'), 'CLG5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-04'), 'CLH5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLH5')
                                          ])
        weights1 = pd.DataFrame(vals, index=widx, columns=["CL0", "CL1"])
        vals = [[1, 0], [0, 1],
                [0.5, 0], [0.5, 0.5], [0, 0.5]]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'COF5'),
                                          (pd.Timestamp('2015-01-03'), 'COG5'),
                                          (pd.Timestamp('2015-01-04'), 'COF5'),
                                          (pd.Timestamp('2015-01-04'), 'COG5'),
                                          (pd.Timestamp('2015-01-04'), 'COH5')
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

    def test_calc_rets_extra_instr_rets(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-03'), 'CLG5'),
                                         (pd.Timestamp('2015-01-03'), 'CLH5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLG5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5'),
                                         (pd.Timestamp('2015-01-06'), 'CLG5')])
        rets = pd.Series([0.1, 0.2, 0.4, 0.05, 0.1, 0.8, 0.01], index=idx)
        vals = [1, 0.5, 0.5, 1]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([0.1, 0.075, 0.8],
                                 index=weights.index.levels[0],
                                 columns=['CL1'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_missing_instr_rets(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5')])
        rets = pd.Series([0.1, 0.2, 0.4], index=idx)
        vals = [1, 0.5, 0.5, 1]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([0.1, pd.np.NaN, 0.4],
                                 index=weights.index.levels[0],
                                 columns=['CL1'])
        assert_frame_equal(wrets, wrets_exp)

    def test_calc_rets_nan_instr_rets(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLF5'),
                                         (pd.Timestamp('2015-01-04'), 'CLG5'),
                                         (pd.Timestamp('2015-01-05'), 'CLG5')])
        rets = pd.Series([pd.np.NaN, pd.np.NaN, 0.1, 0.8], index=idx)
        vals = [1, 0.5, 0.5, 1]
        widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLF5'),
                                          (pd.Timestamp('2015-01-04'), 'CLG5'),
                                          (pd.Timestamp('2015-01-05'), 'CLG5')
                                          ])
        weights = pd.DataFrame(vals, index=widx, columns=['CL1'])
        wrets = util.calc_rets(rets, weights)
        wrets_exp = pd.DataFrame([pd.np.NaN, pd.np.NaN, 0.8],
                                 index=weights.index.levels[0],
                                 columns=['CL1'])
        assert_frame_equal(wrets, wrets_exp)

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

        def extra_trade():
            util.calc_trades(current_contracts, desired_holdings, wts, prices,
                             multipliers)

        self.assertRaises(ValueError, extra_trade)

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

        def format_mismatch():
            util.get_multiplier(wts, ast_mult)

        self.assertRaises(ValueError, format_mismatch)