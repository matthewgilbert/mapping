import pandas as pd
import numpy as np


def read_price_data(files, name_func):
    """
    Convenience function for reading in pricing data from csv files

    Parameters
    ----------
    files: list
        List of strings refering to csv files to read data in from, first
        column should be dates
    name_func: func
        A function to apply to the file strings to infer the instrument name,
        used in the second level of the MultiIndex index.

    Returns
    -------
    A pandas.DataFrame with a pandas.MultiIndex where the top level is
    pandas.Timestamps and the second level is instrument names. Columns are
    given by the csv file columns.
    """

    dfs = []
    for f in files:
        name = name_func(f)
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        df.index = pd.MultiIndex.from_product([df.index, [name]])
        dfs.append(df)

    return pd.concat(dfs, axis=0).sort_index()


def calc_rets(returns, weights):
    """
    Calculate continuous return series for futures instruments. These consist
    of weighted underlying instrument returns, who's weights can vary over
    time.

    Parameters
    ----------
    returns: pandas.Series or dict
        A Series of instrument returns with a MultiIndex where the top level is
        pandas.Timestamps and the second level is instrument names. Values
        correspond to one period instrument returns. returns should be
        available for all for all Timestamps and instruments provided in
        weights. If dict is given this should be a dict of pandas.Series in the
        above format, with assets as keys, e.g. 'CL'
    weights: pandas.DataFrame or dict
        A DataFrame of instrument weights with a MultiIndex where the top level
        contains pandas.Timestamps and the second level is instrument names.
        The columns consist of generic names. If dict is given this should be
        a dict of pandas.Series in the above format, with assets as keys, e.g.
        'CL'

    Returns
    -------
    A pandas.DataFrame of continuous returns for generics. The index is
    pandas.Timestamps and the columns is generic names, corresponding to
    weights.columns or if dicts are given then the columns consist of the dict
    keys concatened with the columns of the values.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-02'), 'CLF5'),
    ...                                  (pd.Timestamp('2015-01-03'), 'CLF5'),
    ...                                  (pd.Timestamp('2015-01-03'), 'CLG5'),
    ...                                  (pd.Timestamp('2015-01-04'), 'CLF5'),
    ...                                  (pd.Timestamp('2015-01-04'), 'CLG5'),
    ...                                  (pd.Timestamp('2015-01-05'), 'CLG5')])
    >>> price = pd.Series([45.63, 45.85, 46.13, 46.05, 46.25, 46.20], index=idx)
    >>> vals = [1, 1/2, 1/2, 1]
    >>> widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLF5'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLG5'),
    ...                                   (pd.Timestamp('2015-01-05'), 'CLG5')])
    >>> weights = pd.DataFrame(vals, index=widx, columns=[0])
    >>> irets = price.groupby(level=-1).pct_change()
    >>> util.calc_rets(irets, weights)
    """  # NOQA

    if not isinstance(returns, dict):
        returns = {"": returns}
    if not isinstance(weights, dict):
        weights = {"": weights}

    grets = []
    cols = []
    for ast in returns:
        wts = weights[ast]
        rets = returns[ast].loc[wts.index]

        for generic in wts.columns:
            # grouby time
            group_rets = (rets * wts.loc[:, generic]).groupby(level=0)
            grets.append(group_rets.apply(pd.DataFrame.sum, skipna=False))

        if ast == "":
            cols.extend(wts.columns.tolist())
        else:
            cols.extend([ast + str(i) for i in wts.columns.tolist()])

    rets = pd.concat(grets, axis=1, keys=cols)
    rets = rets.loc[:, rets.columns.sort_values()]
    return rets


def calc_trades(current_contracts, desired_holdings, weights, prices,
                **kwargs):
    """
    Calculate the number of tradeable contracts for rebalancing from a set
    of current contract holdings to a set of desired generic notional holdings
    based on prevailing prices and mapping from generics to tradeable
    instruments. Differences between current holdings and desired holdings
    are treated as 0. Zero trades are dropped.

    Parameters
    ----------
    current_contracts: pandas.Series
        Series of current number of contracts held for tradeable instruments.
    desired_holdings: pandas.Series
        Series of desired holdings in base notional currency of generics. Index
        is generic contracts, these should be the same generics as in weights.
    weights: pandas.DataFrame
        A pandas.DataFrame of loadings of generic contracts on tradeable
        instruments for a given date. The columns are integers refering to
        generic number indexed from 0, e.g. [0, 1], and the index is strings
        representing instrument names. If dict is given keys should be generic
        instrument names, e.g. 'CL', and values should be pandas.DataFrames of
        loadings. The union of all indexes should be a superset of the
        instruments.index
    prices: pandas.Series
        Series of instrument prices. Index is instrument name and values are
        number of contracts.
    kwargs: key word arguments
        Key word arguments to be passed to to_contracts()

    Returns
    -------
    A pandas.Series of instrument contract trades, lexigraphically sorted.

    Example
    -------
    >>> wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
    ...                    index=["CLX16", "CLZ16", "CLF17"],
    ...                    columns=[0, 1])
    >>> desired_holdings = pd.Series([200000, -50000], index=[0, 1])
    >>> current_contracts = pd.Series([0, 1, 0],
    ...                               index=['CLX16', 'CLZ16', 'CLF17'])
    >>> prices = pd.Series([50.32, 50.41, 50.48],
    ...                    index=['CLX16', 'CLZ16', 'CLF17'])
    >>> multiplier = pd.Series([100, 100, 100],
    ...                        index=['CLX16', 'CLZ16', 'CLF17'])
    >>> trades = util.calc_trades(current_contracts, desired_holdings, wts,
    ...                           prices, multipliers=multiplier)

    """
    if not isinstance(weights, dict):
        weights = {"": weights}

    unmapped_instr = desired_holdings.index
    des_cons = []
    for ast in weights:
        ast_weights = weights[ast]

        # allow weights to be a superset of desired holdings, and make sure
        # every holding has been mapped
        ast_des_hlds = desired_holdings.loc[ast_weights.columns].dropna()
        ast_weights = ast_weights.loc[:, ast_des_hlds.index]
        # drop indexes where all non zero weights were in columns dropped above
        ast_weights = ast_weights.loc[~(ast_weights == 0).all(axis=1)]
        unmapped_instr = unmapped_instr.difference(ast_des_hlds.index)

        instr_des_hlds = ast_des_hlds * ast_weights
        instr_des_hlds = instr_des_hlds.sum(axis=1)
        wprices = prices.loc[instr_des_hlds.index]
        des_cons.append(to_contracts(instr_des_hlds, wprices, **kwargs))

    if len(unmapped_instr) > 0:
        raise KeyError("Unmapped desired_holdings %s. weights must be a "
                       "superset of instruments" % unmapped_instr.tolist())

    des_cons = pd.concat(des_cons, axis=0)

    trades = des_cons.subtract(current_contracts, fill_value=0)
    trades = trades.loc[trades != 0]
    trades = trades.sort_index()
    return trades


def to_notional(instruments, prices, desired_ccy=None, instr_fx=None,
                fx_rates=None, multipliers=None):
    """
    Convert number of instruments to notional value in a desired currency.

    Parameters
    ----------
    instruments: pandas.Series
        Series of instrument holdings. Index is instrument name and values are
        number of contracts.
    prices: pandas.Series
        Series of instrument prices. Index is instrument name and values are
        instrument prices. prices.index should be a superset of
        instruments.index
    desired_ccy: str
        Three letter string representing desired currency to convert notional
        values to, e.g. 'USD'. If None is given currency conversion is ignored.
    instr_fx: pandas.Series
        Series of instrument fx denominations. Index is instrument name and
        values are three letter strings representing the currency the
        instrument is denominated in. instr_fx.index should match prices.index
    fx_rates: pandas.Series
        Series of fx rates used for conversion to desired_ccy. Index is strings
        representing the FX pair, e.g. 'AUDUSD' or 'USDCAD'. Values are the
        corresponding exchange rates.
    multipliers: pandas.Series
        Series of instrument multipliers. Index is instrument name and
        values are the multiplier associated with the contract. If None is
        given the multiplier is assumed to be 1.

    Returns
    -------
    pandas.Series of notional amounts of instruments with Index of instruments
    names
    """

    notionals = _instr_conv(instruments, prices, True, desired_ccy,
                            instr_fx, fx_rates, multipliers)
    return notionals


def to_contracts(instruments, prices, desired_ccy=None, instr_fx=None,
                 fx_rates=None, multipliers=None):
    """
    Convert notional amount of instruments to number of instrument contracts,
    rounding to nearest integer number of contracts.

    Parameters
    ----------
    instruments: pandas.Series
        Series of instrument holdings. Index is instrument name and values are
        notional amount on instrument.
    prices: pandas.Series
        Series of instrument prices. Index is instrument name and values are
        instrument prices. prices.index should be a superset of
        instruments.index
    desired_ccy: str
        Three letter string representing desired currency to convert notional
        values to, e.g. 'USD'. If None is given currency conversion is ignored.
    instr_fx: pandas.Series
        Series of instrument fx denominations. Index is instrument name and
        values are three letter strings representing the currency the
        instrument is denominated in. instr_fx.index should match prices.index
    fx_rates: pandas.Series
        Series of fx rates used for conversion to desired_ccy. Index is strings
        representing the FX pair, e.g. 'AUDUSD' or 'USDCAD'. Values are the
        corresponding exchange rates.
    multipliers: pandas.Series
        Series of instrument multipliers. Index is instrument name and
        values are the multiplier associated with the contract. If None is
        given the multiplier is assumed to be 1. multipliers.index should be a
        superset of instruments.index

    Returns
    -------
    pandas.Series of contract numbers of instruments with Index of instruments
    names
    """

    contracts = _instr_conv(instruments, prices, False, desired_ccy,
                            instr_fx, fx_rates, multipliers)
    contracts = contracts.round()
    contracts = contracts.astype(int)
    return contracts


def _instr_conv(instruments, prices, to_notional, desired_ccy, instr_fx,
                fx_rates, multipliers):

    if desired_ccy:
        prices = prices.loc[instr_fx.index]
        conv_rate = []
        for ccy in instr_fx.values:
            conv_rate.append(_get_fx_conversions(fx_rates, ccy, desired_ccy))
        fx_adj_prices = prices * np.array(conv_rate)
    else:
        fx_adj_prices = prices

    if multipliers is None:
        multipliers = 1

    if to_notional:
        amounts = instruments * fx_adj_prices * multipliers
    else:
        amounts = (instruments / fx_adj_prices) / multipliers

    amounts = amounts.loc[instruments.index]

    return amounts


def _get_fx_conversions(fx_rates, ccy, desired_ccy='USD'):
    # return rate to multiply through by to convert from instrument ccy to
    # desired ccy
    # fx_rates is a series of fx rates with index names of the form AUDUSD,
    # USDCAD, etc. ccy is a st
    ccy_pair1 = ccy + desired_ccy
    ccy_pair2 = desired_ccy + ccy
    if ccy == desired_ccy:
        conv_rate = 1.0
    elif ccy_pair1 in fx_rates:
        conv_rate = fx_rates.loc[ccy_pair1]
    elif ccy_pair2 in fx_rates:
        conv_rate = 1 / fx_rates.loc[ccy_pair2]
    else:
        raise(KeyError(ccy_pair1, ccy_pair2))

    return conv_rate
