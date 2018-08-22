import pandas as pd
import numpy as np
import os


def read_price_data(files, name_func=None):
    """
    Convenience function for reading in pricing data from csv files

    Parameters
    ----------
    files: list
        List of strings refering to csv files to read data in from, first
        column should be dates
    name_func: func
        A function to apply to the file strings to infer the instrument name,
        used in the second level of the MultiIndex index. Default is the file
        name excluding the pathname and file ending,
        e.g. /path/to/file/name.csv -> name

    Returns
    -------
    A pandas.DataFrame with a pandas.MultiIndex where the top level is
    pandas.Timestamps and the second level is instrument names. Columns are
    given by the csv file columns.
    """
    if name_func is None:
        def name_func(x):
            return os.path.split(x)[1].split(".")[0]

    dfs = []
    for f in files:
        name = name_func(f)
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        df.index = pd.MultiIndex.from_product([df.index, [name]],
                                              names=["date", "contract"])
        dfs.append(df)

    return pd.concat(dfs, axis=0, sort=False).sort_index()


def flatten(weights):
    """
    Flatten weights into a long DataFrame.

    Parameters
    ----------
    weights: pandas.DataFrame or dict
        A DataFrame of instrument weights with a MultiIndex where the top level
        contains pandas. Timestamps and the second level is instrument names.
        The columns consist of generic names. If dict is given this should be
        a dict of pandas.DataFrame in the above format, with keys for different
        root generics, e.g. 'CL'

    Returns
    -------
    A long DataFrame of weights, where columns are "date", "contract",
    "generic" and "weight". If a dictionary is passed, DataFrame will contain
    additional colum "key" containing the key value and be sorted according to
    this key value.

    Example
    -------
    >>> import pandas as pd
    >>> import mapping.util as util
    >>> vals = [[1, 0], [0, 1], [1, 0], [0, 1]]
    >>> widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF5'),
    ...                                   (pd.Timestamp('2015-01-03'), 'CLG5'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLG5'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLH5')])
    >>> weights = pd.DataFrame(vals, index=widx, columns=["CL1", "CL2"])
    >>> util.flatten(weights)
    """  # NOQA
    if isinstance(weights, pd.DataFrame):
        wts = weights.stack().reset_index()
        wts.columns = ["date", "contract", "generic", "weight"]
    elif isinstance(weights, dict):
        wts = []
        for key in sorted(weights.keys()):
            wt = weights[key].stack().reset_index()
            wt.columns = ["date", "contract", "generic", "weight"]
            wt.loc[:, "key"] = key
            wts.append(wt)
        wts = pd.concat(wts, axis=0).reset_index(drop=True)
    else:
        raise ValueError("weights must be pd.DataFrame or dict")

    return wts


def unflatten(flat_weights):
    """
    Pivot weights from long DataFrame into weighting matrix.

    Parameters
    ----------
    flat_weights: pandas.DataFrame
        A long DataFrame of weights, where columns are "date", "contract",
        "generic", "weight" and optionally "key". If "key" column is
        present a dictionary of unflattened DataFrames is returned with the
        dictionary keys corresponding to the "key" column and each sub
        DataFrame containing rows for this key.

    Returns
    -------
    A DataFrame or dict of DataFrames of instrument weights with a MultiIndex
    where the top level contains pandas.Timestamps and the second level is
    instrument names. The columns consist of generic names. If dict is returned
    the dict keys correspond to the "key" column of the input.

    Example
    -------
    >>> import pandas as pd
    >>> from pandas import Timestamp as TS
    >>> import mapping.util as util
    >>> long_wts = pd.DataFrame(
    ...         {"date": [TS('2015-01-03')] * 4 + [TS('2015-01-04')] * 4,
    ...          "contract": ['CLF5'] * 2 + ['CLG5'] * 4 + ['CLH5'] * 2,
    ...          "generic": ["CL1", "CL2"] * 4,
    ...          "weight": [1, 0, 0, 1, 1, 0, 0, 1]}
    ...     ).loc[:, ["date", "contract", "generic", "weight"]]
    >>> util.unflatten(long_wts)

    See also: calc_rets()
    """  # NOQA
    if flat_weights.columns.contains("key"):
        weights = {}
        for key in flat_weights.loc[:, "key"].unique():
            flt_wts = flat_weights.loc[flat_weights.loc[:, "key"] == key, :]
            flt_wts = flt_wts.drop(labels="key", axis=1)
            wts = flt_wts.pivot_table(index=["date", "contract"],
                                      columns=["generic"],
                                      values=["weight"])
            wts.columns = wts.columns.droplevel(0)
            weights[key] = wts
    else:
        weights = flat_weights.pivot_table(index=["date", "contract"],
                                           columns=["generic"],
                                           values=["weight"])
        weights.columns = weights.columns.droplevel(0)

    return weights


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
        above format, with keys which are a subset of the keys given in weights
    weights: pandas.DataFrame or dict
        A DataFrame of instrument weights with a MultiIndex where the top level
        contains pandas.Timestamps and the second level is instrument names.
        The columns consist of generic names. If dict is given this should be
        a dict of pandas.DataFrame in the above format, with keys for different
        root generics, e.g. 'CL'

    Returns
    -------
    A pandas.DataFrame of continuous returns for generics. The index is
    pandas.Timestamps and the columns is generic names, corresponding to
    weights.columns

    Examples
    --------
    >>> import pandas as pd
    >>> import mapping.util as util
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
    >>> weights = pd.DataFrame(vals, index=widx, columns=["CL1"])
    >>> irets = price.groupby(level=-1).pct_change()
    >>> util.calc_rets(irets, weights)
    """  # NOQA

    if not isinstance(returns, dict):
        returns = {"": returns}
    if not isinstance(weights, dict):
        weights = {"": weights}

    generic_superset = []
    for root in weights:
        generic_superset.extend(weights[root].columns.tolist())
    if len(set(generic_superset)) != len(generic_superset):
        raise ValueError("Columns for weights must all be unique")

    _check_indices(returns, weights)

    grets = []
    cols = []
    for root in returns:
        root_wts = weights[root]
        root_rets = returns[root]
        for generic in root_wts.columns:
            gnrc_wts = root_wts.loc[:, generic]
            # drop generics where weight is 0, this avoids potential KeyError
            # in later indexing of rets even when ret has weight of 0
            gnrc_wts = gnrc_wts.loc[gnrc_wts != 0]
            rets = root_rets.loc[gnrc_wts.index]
            # groupby time
            group_rets = (rets * gnrc_wts).groupby(level=0)
            grets.append(group_rets.apply(pd.DataFrame.sum, skipna=False))
        cols.extend(root_wts.columns.tolist())

    rets = pd.concat(grets, axis=1, keys=cols).sort_index(axis=1)
    return rets


def _stringify(xs):
    if len(xs) <= 2:
        return repr(xs)
    return '[{!r}, ..., {!r}]'.format(xs[0], xs[-1])


def _check_indices(returns, weights):
    # dictionaries of returns and weights

    # check 1: ensure that all non zero instrument weights have associated
    # returns, see https://github.com/matthewgilbert/mapping/issues/3

    # check 2: ensure that returns are not dropped if reindexed from weights,
    # see https://github.com/matthewgilbert/mapping/issues/8

    if list(returns.keys()) == [""]:
        msg1 = ("'returns.index.get_level_values(0)' must contain dates which "
                "are a subset of 'weights.index.get_level_values(0)'"
                "\nExtra keys: {1}")
        msg2 = ("{0} from the non zero elements of "
                "'weights.loc[:, '{2}'].index' are not in 'returns.index'")
    else:
        msg1 = ("'returns['{0}'].index.get_level_values(0)' must contain "
                "dates which are a subset of "
                "'weights['{0}'].index.get_level_values(0)'"
                "\nExtra keys: {1}")
        msg2 = ("{0} from the non zero elements of "
                "'weights['{1}'].loc[:, '{2}'].index' are not in "
                "'returns['{1}'].index'")

    for root in returns:
        wts = weights[root]
        rets = returns[root]

        dts_rets = rets.index.get_level_values(0)
        dts_wts = wts.index.get_level_values(0)
        # check 1
        if not dts_rets.isin(dts_wts).all():
            missing_dates = dts_rets.difference(dts_wts).tolist()
            raise ValueError(msg1.format(root, _stringify(missing_dates)))
        # check 2
        for generic in wts.columns:
            gnrc_wts = wts.loc[:, generic]
            # drop generics where weight is 0, this avoids potential KeyError
            # in later indexing of rets even when ret has weight of 0
            gnrc_wts = gnrc_wts.loc[gnrc_wts != 0]
            # necessary instead of missing_keys.any() to support MultiIndex
            if not gnrc_wts.index.isin(rets.index).all():
                # as list instead of MultiIndex for legibility when stack trace
                missing_keys = (gnrc_wts.index.difference(rets.index).tolist())
                msg2 = msg2.format(_stringify(missing_keys), root, generic)
                raise KeyError(msg2)


def reindex(prices, index, limit):
    """
    Reindex a pd.Series of prices such that when instrument level returns are
    calculated they are compatible with a pd.MultiIndex of instrument weights
    in calc_rets(). This amount to reindexing the series by an augmented
    version of index which includes the preceding date for the first appearance
    of each instrument. Fill forward missing values with previous price up to
    some limit.

    Parameters
    ----------
    prices: pandas.Series
        A Series of instrument prices with a MultiIndex where the top level is
        pandas.Timestamps and the second level is instrument names.
    index: pandas.MultiIndex
        A MultiIndex where the top level contains pandas.Timestamps and the
        second level is instrument names.
    limt: int
        Number of periods to fill prices forward.

    Returns
    -------
    A pandas.Series of reindexed prices where the top level is
    pandas.Timestamps and the second level is instrument names.

    See also: calc_rets()

    Example
    -------
    >>> import pandas as pd
    >>> from pandas import Timestamp as TS
    >>> import mapping.util as util
    >>> idx = pd.MultiIndex.from_tuples([(TS('2015-01-04'), 'CLF5'),
    ...                                  (TS('2015-01-05'), 'CLF5'),
    ...                                  (TS('2015-01-05'), 'CLH5'),
    ...                                  (TS('2015-01-06'), 'CLF5'),
    ...                                  (TS('2015-01-06'), 'CLH5'),
    ...                                  (TS('2015-01-07'), 'CLF5'),
    ...                                  (TS('2015-01-07'), 'CLH5')])
    >>> prices = pd.Series([100.12, 101.50, 102.51, 103.51, 102.73, 102.15,
    ...                     104.37], index=idx)
    >>> widx = pd.MultiIndex.from_tuples([(TS('2015-01-05'), 'CLF5'),
    ...                                   (TS('2015-01-05'), 'CLH5'),
    ...                                   (TS('2015-01-07'), 'CLF5'),
    ...                                   (TS('2015-01-07'), 'CLH5')])
    >>> util.reindex(prices, widx, limit=0)
    """
    if not index.is_unique:
        raise ValueError("'index' must be unique")

    index = index.sort_values()
    price_dts = prices.sort_index().index.unique(level=0)
    index_dts = index.unique(level=0)

    mask = price_dts < index_dts[0]
    leading_price_dts = price_dts[mask]
    if len(leading_price_dts) == 0:
        raise ValueError("'prices' must have a date preceding first date in "
                         "'index'")
    prev_dts = index_dts.tolist()
    prev_dts.insert(0, leading_price_dts[-1])
    # avoid just lagging to preserve the calendar
    previous_date = dict(zip(index_dts, prev_dts))

    first_instr = index.to_frame(index=False)
    first_instr.columns = ["date", "instrument"]
    first_instr = (
        first_instr.drop_duplicates(subset=["instrument"], keep="first")
    )
    first_instr.loc[:, "prev_date"] = (
        first_instr.loc[:, "date"].apply(lambda x: previous_date[x])
    )
    additional_indices = pd.MultiIndex.from_tuples(
        first_instr.loc[:, ["prev_date", "instrument"]].values.tolist()
    )

    augmented_index = index.union(additional_indices).sort_values()
    prices = prices.reindex(augmented_index)
    if limit != 0:
        prices = prices.groupby(level=1).fillna(method="ffill", limit=limit)
    return prices


def calc_trades(current_contracts, desired_holdings, trade_weights, prices,
                multipliers, **kwargs):
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
        Can pass 0 if all holdings are 0.
    desired_holdings: pandas.Series
        Series of desired holdings in base notional currency of generics. Index
        is generic contracts, these should be the same generics as in
        trade_weights.
    trade_weights: pandas.DataFrame or dict
        A pandas.DataFrame of loadings of generic contracts on tradeable
        instruments **for a given date**. The columns refer to generic
        contracts and the index is strings representing instrument names.
        If dict is given keys should be root generic names, e.g. 'CL', and
        values should be pandas.DataFrames of loadings. The union of all
        columns should be a superset of the desired_holdings.index
    prices: pandas.Series
        Series of instrument prices. Index is instrument name and values are
        number of contracts. Extra instrument prices will be ignored.
    multipliers: pandas.Series
        Series of instrument multipliers. Index is instrument name and
        values are the multiplier associated with the contract.
        multipliers.index should be a superset of mapped desired_holdings
        intruments.
    kwargs: key word arguments
        Key word arguments to be passed to to_contracts()

    Returns
    -------
    A pandas.Series of instrument contract trades, lexigraphically sorted.

    Example
    -------
    >>> import pandas as pd
    >>> import mapping.util as util
    >>> wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
    ...                    index=["CLX16", "CLZ16", "CLF17"],
    ...                    columns=["CL1", "CL2"])
    >>> desired_holdings = pd.Series([200000, -50000], index=["CL1", "CL2"])
    >>> current_contracts = pd.Series([0, 1, 0],
    ...                               index=['CLX16', 'CLZ16', 'CLF17'])
    >>> prices = pd.Series([50.32, 50.41, 50.48],
    ...                    index=['CLX16', 'CLZ16', 'CLF17'])
    >>> multipliers = pd.Series([100, 100, 100],
    ...                        index=['CLX16', 'CLZ16', 'CLF17'])
    >>> trades = util.calc_trades(current_contracts, desired_holdings, wts,
    ...                           prices, multipliers)
    """
    if not isinstance(trade_weights, dict):
        trade_weights = {"": trade_weights}

    generics = []
    for key in trade_weights:
        generics.extend(trade_weights[key].columns)

    if not set(desired_holdings.index).issubset(set(generics)):
        raise ValueError("'desired_holdings.index' contains values which "
                         "cannot be mapped to tradeables.\n"
                         "Received: 'desired_holdings.index'\n {0}\n"
                         "Expected in 'trade_weights' set of columns:\n {1}\n"
                         .format(sorted(desired_holdings.index),
                                 sorted(generics)))

    desired_contracts = []
    for root_key in trade_weights:
        gnrc_weights = trade_weights[root_key]

        subset = gnrc_weights.columns.intersection(desired_holdings.index)
        gnrc_des_hlds = desired_holdings.loc[subset]
        gnrc_weights = gnrc_weights.loc[:, subset]
        # drop indexes where all non zero weights were in columns dropped above
        gnrc_weights = gnrc_weights.loc[~(gnrc_weights == 0).all(axis=1)]

        instr_des_hlds = gnrc_des_hlds * gnrc_weights
        instr_des_hlds = instr_des_hlds.sum(axis=1)
        wprices = prices.loc[instr_des_hlds.index]
        desired_contracts.append(to_contracts(instr_des_hlds, wprices,
                                              multipliers, **kwargs))

    desired_contracts = pd.concat(desired_contracts, axis=0)

    trades = desired_contracts.subtract(current_contracts, fill_value=0)
    trades = trades.loc[trades != 0]
    trades = trades.sort_index()
    return trades


def to_notional(instruments, prices, multipliers, desired_ccy=None,
                instr_fx=None, fx_rates=None):
    """
    Convert number of contracts of tradeable instruments to notional value of
    tradeable instruments in a desired currency.

    Parameters
    ----------
    instruments: pandas.Series
        Series of instrument holdings. Index is instrument name and values are
        number of contracts.
    prices: pandas.Series
        Series of instrument prices. Index is instrument name and values are
        instrument prices. prices.index should be a superset of
        instruments.index otherwise NaN returned for instruments without prices
    multipliers: pandas.Series
        Series of instrument multipliers. Index is instrument name and
        values are the multiplier associated with the contract.
        multipliers.index should be a superset of instruments.index
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

    Returns
    -------
    pandas.Series of notional amounts of instruments with Index of instruments
    names

    Example
    -------
    >>> import pandas as pd
    >>> import mapping.util as util
    >>> current_contracts = pd.Series([-1, 1], index=['CLX16', 'CLZ16'])
    >>> prices = pd.Series([50.32, 50.41], index=['CLX16', 'CLZ16'])
    >>> multipliers = pd.Series([100, 100], index=['CLX16', 'CLZ16'])
    >>> ntln = util.to_notional(current_contracts, prices, multipliers)
    """
    notionals = _instr_conv(instruments, prices, multipliers, True,
                            desired_ccy, instr_fx, fx_rates)
    return notionals


def to_contracts(instruments, prices, multipliers, desired_ccy=None,
                 instr_fx=None, fx_rates=None, rounder=None):
    """
    Convert notional amount of tradeable instruments to number of instrument
    contracts, rounding to nearest integer number of contracts.

    Parameters
    ----------
    instruments: pandas.Series
        Series of instrument holdings. Index is instrument name and values are
        notional amount on instrument.
    prices: pandas.Series
        Series of instrument prices. Index is instrument name and values are
        instrument prices. prices.index should be a superset of
        instruments.index
    multipliers: pandas.Series
        Series of instrument multipliers. Index is instrument name and
        values are the multiplier associated with the contract.
        multipliers.index should be a superset of instruments.index
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
    rounder: function
        Function to round pd.Series contracts to integers, if None default
        pd.Series.round is used.

    Returns
    -------
    pandas.Series of contract numbers of instruments with Index of instruments
    names
    """
    contracts = _instr_conv(instruments, prices, multipliers, False,
                            desired_ccy, instr_fx, fx_rates)
    if rounder is None:
        rounder = pd.Series.round

    contracts = rounder(contracts)
    contracts = contracts.astype(int)
    return contracts


def _instr_conv(instruments, prices, multipliers, to_notional, desired_ccy,
                instr_fx, fx_rates):

    if not instruments.index.is_unique:
        raise ValueError("'instruments' must have unique index")
    if not prices.index.is_unique:
        raise ValueError("'prices' must have unique index")
    if not multipliers.index.is_unique:
        raise ValueError("'multipliers' must have unique index")

    if desired_ccy:
        if not instr_fx.index.is_unique:
            raise ValueError("'instr_fx' must have unique index")
        if not fx_rates.index.is_unique:
            raise ValueError("'fx_rates' must have unique index")
        prices = prices.loc[instr_fx.index]
        conv_rate = []
        for ccy in instr_fx.values:
            conv_rate.append(_get_fx_conversions(fx_rates, ccy, desired_ccy))
        fx_adj_prices = prices * np.array(conv_rate)
    else:
        fx_adj_prices = prices

    if to_notional:
        amounts = instruments * fx_adj_prices * multipliers
    else:
        amounts = (instruments / fx_adj_prices) / multipliers

    amounts = amounts.loc[instruments.index]

    return amounts


def get_multiplier(weights, root_generic_multiplier):
    """
    Determine tradeable instrument multiplier based on generic asset
    multipliers and weights mapping from generics to tradeables.

    Parameters
    ----------
    weights: pandas.DataFrame or dict
        A pandas.DataFrame of loadings of generic contracts on tradeable
        instruments **for a given date**. The columns are integers refering to
        generic number indexed from 0, e.g. [0, 1], and the index is strings
        representing instrument names. If dict is given keys should be generic
        instrument names, e.g. 'CL', and values should be pandas.DataFrames of
        loadings. The union of all indexes should be a superset of the
        instruments.index
    root_generic_multiplier: pandas.Series
        Series of multipliers for generic instruments lexigraphically sorted.
        If a dictionary of weights is given, root_generic_multiplier.index
        should correspond to the weights keys.

    Returns
    -------
    A pandas.Series of multipliers for tradeable instruments.

    Examples
    --------
    >>> import pandas as pd
    >>> import mapping.util as util
    >>> wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
    ...                    index=["CLX16", "CLZ16", "CLF17"],
    ...                    columns=[0, 1])
    >>> ast_mult = pd.Series([1000], index=["CL"])
    >>> util.get_multiplier(wts, ast_mult)
    """
    if len(root_generic_multiplier) > 1 and not isinstance(weights, dict):
        raise ValueError("For multiple generic instruments weights must be a "
                         "dictionary")

    mults = []
    intrs = []
    for ast, multiplier in root_generic_multiplier.iteritems():
        if isinstance(weights, dict):
            weights_ast = weights[ast].index
        else:
            weights_ast = weights.index
        mults.extend(np.repeat(multiplier, len(weights_ast)))
        intrs.extend(weights_ast)

    imults = pd.Series(mults, intrs)
    imults = imults.sort_index()
    return imults


def weighted_expiration(weights, contract_dates):
    """
    Calculate the days to expiration for generic futures, weighted by the
    composition of the underlying tradeable instruments.

    Parameters:
    -----------
    weights: pandas.DataFrame
        A DataFrame of instrument weights with a MultiIndex where the top level
        contains pandas.Timestamps and the second level is instrument names.
        The columns consist of generic names.
    contract_dates: pandas.Series
        Series with index of tradeable contract names and pandas.Timestamps
        representing the last date of the roll as values

    Returns:
    --------
    A pandas.DataFrame with columns of generic futures and index of dates.
    Values are the weighted average of days to expiration for the underlying
    contracts.

    Examples:
    ---------
    >>> import pandas as pd
    >>> import mapping.util as util
    >>> vals = [[1, 0, 1/2, 1/2, 0, 1, 0], [0, 1, 0, 1/2, 1/2, 0, 1]]
    >>> widx = pd.MultiIndex.from_tuples([(pd.Timestamp('2015-01-03'), 'CLF15'),
    ...                                   (pd.Timestamp('2015-01-03'), 'CLG15'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLF15'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLG15'),
    ...                                   (pd.Timestamp('2015-01-04'), 'CLH15'),
    ...                                   (pd.Timestamp('2015-01-05'), 'CLG15'),
    ...                                   (pd.Timestamp('2015-01-05'), 'CLH15')])
    >>> weights = pd.DataFrame({"CL1": vals[0], "CL2": vals[1]}, index=widx)
    >>> contract_dates = pd.Series([pd.Timestamp('2015-01-20'),
    ...                             pd.Timestamp('2015-02-21'),
    ...                             pd.Timestamp('2015-03-20')],
    ...                            index=['CLF15', 'CLG15', 'CLH15'])
    >>> util.weighted_expiration(weights, contract_dates)
    """  # NOQA
    cols = weights.columns
    weights = weights.reset_index(level=-1)
    expiries = contract_dates.to_dict()
    weights.loc[:, "expiry"] = weights.iloc[:, 0].apply(lambda x: expiries[x])
    diffs = (pd.DatetimeIndex(weights.expiry)
             - pd.Series(weights.index, weights.index)).apply(lambda x: x.days)
    weights = weights.loc[:, cols]
    wexp = weights.mul(diffs, axis=0).groupby(level=0).sum()
    return wexp


def _get_fx_conversions(fx_rates, ccy, desired_ccy):
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
        raise ValueError("Cannot convert from {0} to {1} with any of "
                         "rates:\n{2}".format(ccy, desired_ccy, fx_rates))

    return conv_rate
