import pandas as pd
import numpy as np
import cvxpy
from pandas.tseries.offsets import BDay


def roller(timestamps, contract_dates, get_weights, **kwargs):
    """
    Calculate weight allocations to tradeable instruments for generic futures
    at a set of timestamps.

    Paramters
    ---------
    timestamps: pd.DatetimeIndex
        Set of timestamps to calculate weights for
    contract_dates: pandas.Series
        Series with index of tradeable contract names and pandas.Timestamps
        representing the last date of the roll as values
    get_weights: function
        A function which takes in a timestamp, contract_dates and **kwargs and
        returns a pandas.DataFrame of loadings of generic contracts on
        tradeable instruments for a given date. The columns are integers
        refering to generic number indexed from 0, e.g. [0, 1], and the index
        is a MultiIndex where the top level is the date represented by a
        pandas.Timestamp and the second level are strings representing
        instrument names.
    kwargs: keyword arguments
        Arguements to pass to get_weights

    Return
    ------
    A pandas.DataFrame with integer columns of generics starting from 0 and a
    MultiIndex of date and contract. Values represent weights on tradeables for
    each generic

    Examples
    --------
    >>> cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
    >>> idx = [-2, -1, 0]
    >>> trans = pd.DataFrame([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5],
    ...                       [0.0, 1.0, 0.0, 1.0]], index=idx, columns=cols)
    >>> contract_dates = pd.Series([pd.Timestamp('2016-10-20'),
    ...                             pd.Timestamp('2016-11-21'),
    ...                             pd.Timestamp('2016-12-20')],
    ...                            index=['CLX16', 'CLZ16', 'CLF17'])
    >>> ts = pd.DatetimeIndex([pd.Timestamp('2016-10-18'),
    ...                        pd.Timestamp('2016-10-19'),
    ...                        pd.Timestamp('2016-10-19')])
    >>> wts = mappings.roller(ts, contract_dates, mappings.static_transition,
    ...                       transition=trans)
    """
    timestamps = timestamps.sort_values()
    contract_dates = contract_dates.sort_values()
    weights = []
    for ts in timestamps:
        contract_dates = contract_dates.loc[contract_dates >= ts]
        weights.append(get_weights(ts, contract_dates, **kwargs))

    weights = pd.concat(weights, axis=0)
    weights = weights.sort_index()
    weights = weights.astype(float)
    return weights


def static_transition(timestamp, contract_dates, transition):
    """
    Return weights to tradeable instruments for a given date based on a
    transition DataFrame which indicates how to role through the role period.

    Parameters
    ----------
    timestamp: pandas.DatetimeIndex
        The timestamp to return instrument weights for
    contract_dates: pandas.Series
        Series with index of tradeable contract names and pandas.Timestamps
        representing the last date of the roll as values
    transition: pandas.DataFrame
        A DataFrame with a index of integers representing business day offsets
        from the last roll date and a column which is a MultiIndex where the
        top level is integers representing generic instruments starting from 0
        and the second level is ['front', 'back'] which refer to the front
        month contract and the back month contract of the role. Note that for
        different generics, e.g. CL1, CL2, the front and back month
        contract during a roll would refer to different underlying instruments.
        The values represent the fraction of the roll on each day during the
        roll period. The first row of the transition period should be
        completely allocated to the front contract and the last row should be
        completely allocated to the back contract.

    Returns
    -------
    A pandas.DataFrame of loadings of generic contracts on tradeable
    instruments for a given date. The columns are integers refering to generic
    number indexed from 0, e.g. [0, 1], and the index is a MultiIndex where the
    top level is the date represented by a pandas.Timestamp and the second
    level are strings representing instrument names.

    Examples
    --------
    >>> cols = pd.MultiIndex.from_product([[0, 1], ['front', 'back']])
    >>> idx = [-2, -1, 0]
    >>> transition = pd.DataFrame([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5],
    ...                            [0.0, 1.0, 0.0, 1.0]],
    ...                           index=idx, columns=cols)
    >>> contract_dates = pd.Series([pd.Timestamp('2016-10-20'),
    ...                             pd.Timestamp('2016-11-21'),
    ...                             pd.Timestamp('2016-12-20')],
    ...                            index=['CLX16', 'CLZ16', 'CLF17'])
    >>> ts = pd.Timestamp('2016-10-19')
    >>> wts = mappings.static_transition(ts, contract_dates, transition)
    """
    contract_dates = contract_dates.loc[contract_dates >= timestamp]
    front_expiry_dt = contract_dates.iloc[0]
    transition = transition.copy()
    new_idx = [BDay(i) + front_expiry_dt for i in transition.index]
    transition.index = new_idx

    weights = transition.reindex([timestamp], method='bfill').loc[timestamp]
    new_idx2 = []
    for gen_num, position in weights.index.tolist():
        if position == "front":
            val = gen_num
        elif position == "back":
            val = gen_num + 1
        else:
            raise ValueError("transition.columns must contain "
                             "'front' or 'back'")
        new_idx2.append((gen_num, contract_dates.index[val]))

    weights.index = pd.MultiIndex.from_tuples(new_idx2)
    weights = weights.reset_index()
    weights.columns = ["generic", "contract", "weight"]
    weights["date"] = timestamp

    weights = weights.groupby(['contract', 'date']).filter(lambda x: x.weight.any())  # NOQA
    weights = weights.pivot_table(index=['date', 'contract'],
                                  columns=['generic'], values='weight',
                                  fill_value=0)
    weights = weights.astype(float)
    return weights


def to_generics(instruments, weights):
    """
    Map tradeable instruments to generics given weights and tradeable
    instrument holdings. This is solving the equation Ax = b where A is the
    weights, and b is the instrument holdings. When Ax = b has no solution we
    solve for x' such that Ax' is closest to b in the least squares sense with
    the additional constraint that sum(x') = sum(instruments).

    Scenarios with exact solutions and non exact solutions are depicted below

    +------------+-----+-----+ Instruments
    | contract   |   0 |   1 | ------------------------------------
    |------------+-----+-----| Scenario 1 | Scenario 2 | Scenario 3
    | CLX16      | 0.5 | 0   | 10         | 10         | 10
    | CLZ16      | 0.5 | 0.5 | 20         | 20         | 25
    | CLF17      | 0   | 0.5 | 10         | 11         | 11
    +------------+-----+-----+

    In scenario 1 the solution is given by x = [20, 20], in scenario 2 the
    solution is given by x = [19.5, 21.5], and in scenario 3 the solution is
    given by x = [22, 24].

    NOTE: Integer solutions are not guruanteed, as demonstrated above.

    Parameters
    ----------
    instruments: pandas.Series
        Series of tradeable instrument holdings where the index is the name of
        the tradeable instrument and the value is the number of that instrument
        held.
    weights: pandas.DataFrame or dict
        A pandas.DataFrame of loadings of generic contracts on tradeable
        instruments for a given date. The columns are integers refering to
        generic number indexed from 0, e.g. [0, 1], and the index is strings
        representing instrument names. If dict is given keys should be generic
        instrument names, e.g. 'CL', and values should be pandas.DataFrames of
        loadings. The union of all indexes should be a superset of the
        instruments.index

    Returns
    -------
    A pandas.Series where the index is the generic and the value is the number
    of contracts.

    Examples
    --------
    >>> wts = pd.DataFrame([[0.5, 0], [0.5, 0.5], [0, 0.5]],
    ...                    index=["CLX16", "CLZ16", "CLF17"],
    ...                    columns=[0, 1])
    >>> instrs = pd.Series([10, 20, 10], index=["CLX16", "CLZ16", "CLF17"])
    >>> generics = mappings.to_generics(instrs, wts)
    """
    if not isinstance(weights, dict):
        weights = {"": weights}

    allocations = []
    unmapped_instr = instruments.index
    for key in weights:
        w = weights[key]
        # may not always have instrument holdings for a set of weights so allow
        # weights to be a superset of instruments, drop values where no
        # holdings
        winstrs = instruments.loc[w.index].dropna()
        w = w.loc[winstrs.index]

        unmapped_instr = unmapped_instr.difference(winstrs.index)

        A = w.values
        b = winstrs.values
        x = cvxpy.Variable(A.shape[1])
        constrs = [cvxpy.sum_entries(x) == np.sum(b)]
        obj = cvxpy.Minimize(cvxpy.sum_squares(A * x - b))
        prob = cvxpy.Problem(obj, constrs)
        prob.solve()

        vals = np.array(x.value).squeeze()
        if key == "":
            idx = w.columns.tolist()
        else:
            idx = [key + str(i) for i in w.columns.tolist()]
        allocations.append(pd.Series(vals, index=idx))

    if len(unmapped_instr) > 0:
        raise KeyError("Unmapped instruments %s. weights must be a superset of"
                       " instruments" % unmapped_instr.tolist())

    allocations = pd.concat(allocations, axis=0)
    return allocations
