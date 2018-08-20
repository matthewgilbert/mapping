import matplotlib.pyplot as plt
import pandas as pd


def plot_composition(df, intervals, axes=None):
    """
    Plot time series of generics and label underlying instruments which
    these series are composed of.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame of time series to be plotted. Each column is a generic time
        series.
    intervals: pd.DataFrame
        A DataFrame including information for when each contract is used in the
        generic series.
        Columns are['contract', 'generic', 'start_date', 'end_date']
    axes: list
        List of matplotlib.axes.Axes

    Example
    -------
    >>> import mapping.plot as mplot
    >>> import pandas as pd
    >>> from pandas import Timestamp as TS
    >>> idx = pd.date_range("2017-01-01", "2017-01-15")
    >>> rets_data = pd.np.random.randn(len(idx))
    >>> rets = pd.DataFrame({"CL1": rets_data, "CL2": rets_data}, index=idx)
    >>> intervals = pd.DataFrame(
    ...   [(TS("2017-01-01"), TS("2017-01-05"), "2017_CL_F", "CL1"),
    ...    (TS("2017-01-05"), TS("2017-01-15"), "2017_CL_G", "CL1"),
    ...    (TS("2017-01-01"), TS("2017-01-12"), "2017_CL_G", "CL2"),
    ...    (TS("2017-01-10"), TS("2017-01-15"), "2017_CL_H", "CL2")],
    ...  columns=["start_date", "end_date", "contract", "generic"])
    >>> mplot.plot_composition(rets, intervals)
    """

    generics = df.columns
    if (axes is not None) and (len(axes) != len(generics)):
        raise ValueError("If 'axes' is not None then it must be the same "
                         "length as 'df.columns'")

    if axes is None:
        _, axes = plt.subplots(nrows=len(generics), ncols=1)
        if len(generics) == 1:
            axes = [axes]

    for ax, generic in zip(axes, generics):
        ax.plot(df.loc[:, generic], label=generic)
        # no legend line to avoid clutter
        ax.legend(loc='center right', handlelength=0)
        dates = intervals.loc[intervals.loc[:, "generic"] == generic,
                              ["start_date", "end_date", "contract"]]
        date_ticks = set(
            dates.loc[:, "start_date"].tolist() +
            dates.loc[:, "end_date"].tolist()
        )
        xticks = [ts.toordinal() for ts in date_ticks]
        xlabels = [ts.strftime("%Y-%m-%d") for ts in date_ticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        y_top = ax.get_ylim()[1]
        count = 0
        # label and colour each underlying
        for _, dt1, dt2, instr in dates.itertuples():
            if count % 2:
                fc = "b"
            else:
                fc = "r"
            count += 1
            ax.axvspan(dt1, dt2, facecolor=fc, alpha=0.2)
            x_mid = dt1 + (dt2 - dt1) / 2
            ax.text(x_mid, y_top, instr, rotation=45)

    return axes


def intervals(weights):
    """
    Extract intervals where generics are composed of different tradeable
    instruments.

    Parameters
    ----------
    weights: DataFrame or dict
        A DataFrame or dictionary of DataFrames with columns representing
        generics and a MultiIndex of date and contract. Values represent
        weights on tradeables for each generic.

    Returns
    -------
    A DataFrame with [columns]
    ['contract', 'generic', 'start_date', 'end_date']

    """
    intrvls = []
    if isinstance(weights, dict):
        for root in weights:
            wts = weights[root]
            intrvls.append(_intervals(wts))
        intrvls = pd.concat(intrvls, axis=0)
    else:
        intrvls = _intervals(weights)
    intrvls = intrvls.reset_index(drop=True)
    return intrvls


def _intervals(weights):
    # since weights denote weightings for returns, not holdings. To determine
    # the previous day we look at the index since lagging would depend on the
    # calendar. As a kludge we omit the first date since impossible to
    # know
    dates = weights.index.get_level_values(0)
    date_lookup = dict(zip(dates[1:], dates[:-1]))

    weights = weights.stack()
    weights.index.names = ["date", "contract", "generic"]
    weights.name = "weight"
    weights = weights.reset_index()
    grps = (weights.loc[weights.weight != 0, :].drop("weight", axis=1)
            .groupby(["contract", "generic"]))

    intrvls = pd.concat([
        grps.min().rename({"date": "start_date"}, axis=1),
        grps.max().rename({"date": "end_date"}, axis=1)],
       axis=1)

    intrvls = intrvls.reset_index().sort_values(["generic", "start_date"])
    # start date should be the previous trading day since returns are from
    # t-1 to t therefore position established at time t-1
    intrvls.loc[:, "start_date"] = (
        intrvls.loc[:, "start_date"].apply(lambda x: date_lookup.get(x, x))
    )
    intrvls = intrvls.loc[:, ['contract', 'generic', 'start_date', 'end_date']]
    return intrvls
