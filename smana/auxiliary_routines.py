import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt
from statsmodels.tsa.seasonal import STL


def nan_presence_counter(df: pd.DataFrame, scan_column: str, datetime_column: str, resolution: timedelta
                         ) -> pd.DataFrame:
    """Identify NaN-value sequences in a series.

    This function analyzes the column `scan_column` of the
    chronologically ordered dataframe `df` to get information about
    the sequences of NaN values present in it. For each sequence,
    the first and the last datetime and the length of itself
    (`first_dt`, `last_dt` and `nan_counter`) are collected in the
    returned dataframe; the column `datetime_column` is used to
    compute each length.

    Parameters
    ----------
    df: DataFrame
        dataframe to be analyzed.
    scan_column: string
        label of the column to be analyzed.
    datetime_column: string
        label of the datetime column.
    resolution: timedelta
        dataframe resolution.

    Returns
    -------
    df_nan_list: DataFrame
        dataframe collecting information on NaN sequences.
    """
    current_df = df.copy()

    nan_list = []
    current_df = current_df[current_df[scan_column].isna()].copy()
    len_curr_df = len(current_df)

    i = 0
    while i < len_curr_df:
        first_index = i
        j = i + 1
        last_index = -1
        while (j < len_curr_df) & (last_index == -1):
            if current_df[datetime_column].iloc[j] != current_df[datetime_column].iloc[j - 1] + resolution:
                last_index = j - 1
            else:
                j += 1
        if last_index == -1:
            last_index = len_curr_df - 1

        ins_first_dt = current_df[datetime_column].iloc[first_index].to_pydatetime()
        ins_last_dt = current_df[datetime_column].iloc[last_index].to_pydatetime()
        ins_delta = ins_last_dt-ins_first_dt
        nan_list = nan_list + [(ins_first_dt, ins_last_dt,
                                int(1 + ins_delta.total_seconds() / resolution.total_seconds()))]

        i = last_index + 1

    df_nan_list = pd.DataFrame(nan_list, columns=["first_dt", "last_dt", "nan_counter"])

    return df_nan_list


def evaluate_holes_distances(df_nan_list: pd.DataFrame, start_datetime: dt.datetime,
                             end_datetime: dt.datetime, resolution: timedelta) -> None:
    """Rank NaN-value sequences.

    This function provides for each NaN-value sequence (saying S),
    stored in `df_nan_list` dataframe, some information about the
    distance of (S) from the nearest sequences and the ratio between
    the distance of (S) from the others and the length of (S).

    Parameters
    ----------
    df_nan_list: DataFrame
        dataframe collecting information on NaN positions.
    start_datetime: datetime
        first timestamp present in original data.
    end_datetime: datetime
        last timestamp present in original data.
    resolution: timedelta
        data granularity expressed as a timedelta object.

    Returns
    -------

    """

    n_seconds_in_a_week = 60 * 60 * 24 * 7
    n_resolution_seconds = int(resolution.total_seconds())

    len_nan = len(df_nan_list)

    if len_nan == 0:
        return

    df_nan_list['left_distance'] = pd.Series(0, index=df_nan_list.index)
    df_nan_list['right_distance'] = pd.Series(0, index=df_nan_list.index)

    if len_nan == 1:
        df_nan_list.left_distance.iat[0] = (df_nan_list.first_dt.iat[0] - start_datetime).total_seconds()
        df_nan_list.right_distance.iat[0] = (end_datetime - df_nan_list.last_dt.iat[0]).total_seconds()

    else:
        common_values = ((np.array(df_nan_list.first_dt.iloc[1:len_nan]
                                   ) - np.array(df_nan_list.last_dt.iloc[0:len_nan-1]
                                                )) / 1e9).astype('int') - n_resolution_seconds

        df_nan_list['left_distance'] = pd.Series(
            [int((df_nan_list.first_dt.iat[0] - start_datetime).total_seconds())] + list(common_values),
            dtype=int, index=df_nan_list.index)
        df_nan_list['right_distance'] = pd.Series(
            list(common_values) + [int((end_datetime - df_nan_list.last_dt.iat[-1]).total_seconds())],
            dtype=int, index=df_nan_list.index)

    df_nan_list['mutual_distance'] = df_nan_list['left_distance'] + df_nan_list['right_distance']
    df_nan_list['left_weeks'] = df_nan_list['left_distance'] // n_seconds_in_a_week
    df_nan_list['right_weeks'] = df_nan_list['right_distance'] // n_seconds_in_a_week

    df_nan_list['total_weeks'] = df_nan_list.left_weeks + df_nan_list.right_weeks
    df_nan_list['symmetrical_weeks'] = df_nan_list[['left_weeks', 'right_weeks']].min(axis=1)
    df_nan_list['long_side_weeks'] = df_nan_list[['left_weeks', 'right_weeks']].max(axis=1)
    df_nan_list['ratio'] = df_nan_list.mutual_distance / df_nan_list.nan_counter

    return


def get_weekly_index(timestamp: dt.datetime, resolution: timedelta) -> int:
    """ Return the weekly integer index of a timestamp.

    Parameters
    ----------
    timestamp: datetime
        timestamp to be processed.
    resolution: timedelta
        dataframe resolution.

    Returns
    -------
    weekly_index: integer
        timestamp weekly index.
    """

    weekly_index = int(((timestamp.weekday() * timedelta(days=1).total_seconds()
                         ) + (timestamp.hour * timedelta(hours=1).total_seconds()
                              ) + (timestamp.minute * timedelta(minutes=1).total_seconds()
                                   ) + timestamp.second
                        ) / resolution.total_seconds())

    return weekly_index


def get_daily_index(timestamp: dt.datetime, resolution: timedelta) -> int:
    """ Return the daily integer index of a timestamp.

    Parameters
    ----------
    timestamp: datetime
        timestamp to be processed.
    resolution: timedelta
        dataframe resolution.

    Returns
    -------
    daily_index: integer
        timestamp daily index.
    """

    daily_index = int(((timestamp.hour * timedelta(hours=1).total_seconds()
                        ) + (timestamp.minute * timedelta(minutes=1).total_seconds()
                             ) + timestamp.second
                       ) / resolution.total_seconds())

    return daily_index


def get_seasonal_component(df: pd.DataFrame, scan_column: str, datetime_column: str,
                           resolution: timedelta, left_bound_list: list) -> np.ndarray:
    """Get seasonal component of a time series.

    This function extracts the weekly periodic component of the time
    series `scan_column` of the dataframe `df`, by means of STL
    decomposition. The extraction concerns only each one of the
    7-days sequences with the first timestamp present in the
    `left_bound_list` list.

    Parameters
    ----------
    df: DataFrame
        dataframe to be analyzed.
    scan_column: string
        label of the column to be analyzed.
    datetime_column: string
        label of the datetime column.
    resolution: timedelta
        dataframe resolution.
    left_bound_list: list
        list of the first datetime of the 7-days sequences.

    Returns
    -------
    overall_weekly_component: ndarray
        seasonal component of the time series.
    """

    n_components_in_a_week = int(timedelta(weeks=1).total_seconds() / resolution.total_seconds())
    i = 0
    sequences = [[left_bound_list[0]]]
    j = 1
    while j < len(left_bound_list):
        if left_bound_list[j] != left_bound_list[j-1] + timedelta(weeks=1):
            sequences.append([left_bound_list[j]])
            i += 1
        else:
            sequences[i].append(left_bound_list[j])
        j += 1

    weights = [len(x) for x in sequences]
    sequences_first_element = [x[0] for x in sequences]

    # Try to consider only sequences which have weight greater than or equal to 2 (weeks)
    long_sequences_indices = [i for i, v in enumerate(weights) if v >= 2]
    if len(long_sequences_indices) > 0:
        weights = [weights[i] for i in long_sequences_indices]
        sequences_first_element = [sequences_first_element[i] for i in long_sequences_indices]

    is_first_sequence = True
    overall_data = None
    for seq_index in range(len(sequences_first_element)):
        sub_df = df[(df[datetime_column] >= sequences_first_element[seq_index]) & (
                df[datetime_column] < sequences_first_element[seq_index] + timedelta(weeks=weights[seq_index]))
                    ][scan_column].values

        sequence_seasonal_component = STL(sub_df, period=n_components_in_a_week).fit().seasonal.reshape(
            (-1, n_components_in_a_week))
        sequence_weekly_component = np.mean(sequence_seasonal_component, axis=0)

        shift = get_weekly_index(timestamp=sequences_first_element[seq_index], resolution=resolution)
        sequence_weekly_component = np.roll(sequence_weekly_component, shift)
        sequence_weekly_component = sequence_weekly_component - sequence_weekly_component.mean()

        if is_first_sequence:
            overall_data = sequence_weekly_component
            is_first_sequence = False
        else:
            overall_data = np.concatenate((overall_data, sequence_weekly_component), axis=0)

    overall_weekly_component = np.average(overall_data.reshape((-1, n_components_in_a_week)),
                                          weights=weights, axis=0)
    overall_weekly_component = overall_weekly_component - overall_weekly_component.mean()

    return overall_weekly_component


def substitute_holes(df: pd.DataFrame, scan_column: str, datetime_column: str, resolution: timedelta,
                     df_nan_list: pd.DataFrame, seasonal_comp: np.ndarray, trend_approx_size: int) -> pd.DataFrame:
    """Reconstruct each missing value using STL decomposition.

    This function replaces each NaN-value sequence present in
    `df_nan_list` by adding the weekly periodicity `seasonal_comp`
    and an approximation of the trend component, based on the first
    `trend_approx_size` values which time-wise precede or follow
    the sequence itself. The `df` dataframe is updated and returned.

    Parameters
    ----------
    df: DataFrame
        dataframe to be analyzed.
    scan_column: string
        label of the column to be analyzed.
    datetime_column: string
        label of the datetime column.
    resolution: timedelta
        dataframe resolution.
    df_nan_list: DataFrame
        dataframe collecting information on NaN positions.
    seasonal_comp: ndarray
        weekly seasonal component of the time series.
    trend_approx_size: integer
        trend approximation parameter.

    Returns
    -------
    df: DataFrame
        Updated dataframe.
    """

    len_nan = len(df_nan_list)

    i = 0
    j = 1
    if df[datetime_column].iat[0] == df_nan_list.first_dt.iat[0]:
        trend = np.zeros(min(trend_approx_size, len(df)-df_nan_list.nan_counter.iat[0]))
        for t in range(len(trend)):
            local_index = df_nan_list.nan_counter.iat[0] + t
            trend[t] = df[scan_column].iat[local_index] - seasonal_comp[
                get_weekly_index(timestamp=df[datetime_column].iat[local_index], resolution=resolution)]

        trend_avg = np.nanmean(trend)

        for t in range(df_nan_list.nan_counter.iat[0]):
            df[scan_column].iat[t] = trend_avg + seasonal_comp[get_weekly_index(timestamp=df[datetime_column].iat[t],
                                                                                resolution=resolution)]
        i = 1
        j = df_nan_list.nan_counter.iat[0]

    if df[datetime_column].iat[-1] == df_nan_list.last_dt.iat[-1]:
        last_is_nan = -1
    else:
        last_is_nan = 0

    while (j < len(df)-1) & (i < len_nan + last_is_nan):

        if df[datetime_column].iat[j] == df_nan_list.first_dt.iat[i]:

            trend_pre = np.zeros(min(trend_approx_size, j))
            for t in range(len(trend_pre)):
                local_index = j + t - len(trend_pre)
                trend_pre[t] = df[scan_column].iat[local_index] - seasonal_comp[
                    get_weekly_index(timestamp=df[datetime_column].iat[local_index], resolution=resolution)]
            trend_pre_avg = np.nanmean(trend_pre)

            trend_post = np.zeros(min(trend_approx_size, len(df)-j-df_nan_list.nan_counter.iat[i]))
            for t in range(len(trend_post)):
                local_index = j + df_nan_list.nan_counter.iat[i] + t
                trend_post[t] = df[scan_column].iat[local_index] - seasonal_comp[
                    get_weekly_index(timestamp=df[datetime_column].iat[local_index], resolution=resolution)]
            trend_post_avg = np.nanmean(trend_post)

            t_coeff = (trend_post_avg - trend_pre_avg) / (
                    (len(trend_pre) - 1.0) / 2.0 + df_nan_list.nan_counter.iat[i] + (len(trend_post) - 1.0) / 2.0 + 1.0
                                                          )

            for t in range(df_nan_list.nan_counter.iat[i]):
                temp_trend = t_coeff * (t + (len(trend_pre) + 1.0)/2.0) + trend_pre_avg
                df[scan_column].iat[j+t] = temp_trend + seasonal_comp[
                    get_weekly_index(timestamp=df[datetime_column].iat[j+t], resolution=resolution)]

            j += df_nan_list.nan_counter.iat[i]
            i += 1
        else:
            j += 1

    if last_is_nan == -1:
        flag = True
        while flag:
            if df[datetime_column].iat[j] == df_nan_list.first_dt.iat[-1]:

                trend = np.zeros(min(trend_approx_size, j))
                for t in range(len(trend)):
                    local_index = j + t - len(trend)
                    trend[t] = df[scan_column].iat[local_index] - seasonal_comp[
                        get_weekly_index(timestamp=df[datetime_column].iat[local_index], resolution=resolution)]
                trend_avg = np.nanmean(trend)

                for t in range(df_nan_list.nan_counter.iat[i]):
                    df[scan_column].iat[j + t] = trend_avg + seasonal_comp[
                        get_weekly_index(timestamp=df[datetime_column].iat[j+t], resolution=resolution)]

                flag = False
            else:
                j += 1

    return df


def find_good_weeks_left_boundary(df_nan_list: pd.DataFrame, start_datetime: dt.datetime,
                                  end_datetime: dt.datetime, resolution: timedelta) -> list:
    """Search for 7-days data sequences without missing value.

    This function detects disjoint 7-days sequences of valid
    values within the time series; for each sequence, the first
    datetime is stored in the returned list.

    Parameters
    ----------
    df_nan_list: DataFrame
        dataframe collecting information on NaN positions.
    start_datetime: datetime
        first timestamp present in original data.
    end_datetime: datetime
        last timestamp present in original data.
    resolution: timedelta
        dataframe resolution.

    Returns
    -------
    g_w: list
        list of the first datetime of the 7-days sequences.
    """

    len_df = len(df_nan_list)
    g_w = []
    i = 0
    if len_df == 0:
        while end_datetime >= start_datetime + timedelta(weeks=i) + timedelta(weeks=1) - resolution:
            g_w.append(start_datetime + timedelta(weeks=i))
            i += 1
    else:
        while df_nan_list.first_dt.iloc[0] > start_datetime + timedelta(weeks=i) + timedelta(weeks=1) - resolution:
            g_w.append(start_datetime + timedelta(weeks=i))
            i += 1

        for j in range(len_df-1):
            i = 0
            while df_nan_list.first_dt.iloc[j+1] > df_nan_list.last_dt.iloc[j] + timedelta(weeks=i
                                                                                           ) + timedelta(weeks=1):
                temp = df_nan_list.last_dt.iloc[j] + timedelta(weeks=i) + resolution
                g_w.append(temp.to_pydatetime())
                i += 1

        i = 0
        while end_datetime >= df_nan_list.last_dt.iloc[-1] + timedelta(weeks=i) + timedelta(weeks=1):
            temp = df_nan_list.last_dt.iloc[-1] + resolution + timedelta(weeks=i)
            g_w.append(temp.to_pydatetime())
            i += 1

    return g_w


def interpolate_holes(df: pd.DataFrame, scan_column: str, datetime_column: str, resolution: timedelta,
                      df_nan_list: pd.DataFrame, nonnegative_constraint: bool,
                      auxiliary_restored_column_name: str) -> pd.DataFrame:
    """Interpolate missing-value sequences.

    This function linearly interpolates missing-value sequences
    which are stored in `df_nan_list`; the `df` dataframe is
    updated and returned.

    Parameters
    ----------
    df: DataFrame
        dataframe to be analyzed.
    scan_column: string
        label of the column to be analyzed.
    datetime_column: string
        label of the datetime column.
    resolution: timedelta
        dataframe resolution.
    df_nan_list: DataFrame
        dataframe collecting information on NaN positions.
    nonnegative_constraint: boolean
        set to True to check and repair negative restored values.
    auxiliary_restored_column_name: basestring
        auxiliary column label.

    Returns
    -------
    df: DataFrame
        Updated data dataframe.
    """

    for curr_row in df_nan_list.itertuples(index=False):
        sub_df = df[(df[datetime_column] >= curr_row.first_dt - resolution
                     ) & (df[datetime_column] <= curr_row.last_dt + resolution)].copy()
        sub_df[scan_column] = sub_df[scan_column].interpolate(method='linear',
                                                              limit_direction='both')
        # Check for any negative reconstructed components
        if nonnegative_constraint:
            sub_df.loc[(sub_df[scan_column] < 0.0) & (sub_df[auxiliary_restored_column_name] == 1),
                       scan_column] = 0.0

        for curr_timestamp in sub_df[datetime_column]:
            df.loc[df[datetime_column] == curr_timestamp,
                   scan_column] = sub_df[sub_df[datetime_column] == curr_timestamp][scan_column].iat[0]

    return df


def get_daily_periodic_component(df: pd.DataFrame, scan_column: str, date_column: str, resolution: timedelta) -> dict:
    """Get daily component of a time series.

    This function extracts the daily periodic component of the time
    series `scan_column` of the dataframe `df`.

    Parameters
    ----------
    df: DataFrame
        dataframe to be analyzed.
    scan_column: string
        label of the column to be analyzed.
    date_column: string
        label of the date column.
    resolution: timedelta
        dataframe resolution.

    Returns
    -------
    output_dict: dictionary
        decomposition data.
    """
    is_first_day = True
    n_components_in_a_single_day = int(timedelta(days=1).total_seconds() / resolution.total_seconds())
    input_days = list(df[date_column].unique())
    n_days = len(input_days)
    cumulative_trend = 0.0
    data = None
    for curr_day in input_days:
        curr_values = df[df[date_column] == curr_day][scan_column].values
        curr_trend = np.mean(curr_values)
        curr_seasonal = curr_values - curr_trend

        cumulative_trend += curr_trend
        if is_first_day:
            data = curr_seasonal
            is_first_day = False
        else:
            data = np.concatenate((data, curr_seasonal), axis=0)

    daily_seasona_component = np.mean(data.reshape((-1, n_components_in_a_single_day)), axis=0)
    daily_seasona_component = daily_seasona_component - daily_seasona_component.mean()

    mean_trend = cumulative_trend / n_days

    output_dict = {'avg_trend': mean_trend, 'seasonal': daily_seasona_component}

    return output_dict


def midweek_holidays_replacement(df: pd.DataFrame, scan_column: str, datetime_column: str, resolution: timedelta,
                                 auxiliary_weekday_id_column: str, auxiliary_restored_column_name: str,
                                 week_holiday_int: int, nonnegative_constraint: bool,
                                 max_good_weeks: int, min_good_weeks: int,
                                 midweek_holidays_datetime_list: list) -> pd.DataFrame:
    """Restore midweek holidays.

    Parameters
    ----------
    df: DataFrame
        data dataframe.
    scan_column: string
        label of the column to be analyzed.
    datetime_column: string
        label of the datetime column; if unspecified, dataframe
        index is used.
    resolution: timedelta
        dataframe resolution.
    auxiliary_weekday_id_column: basestring
        auxiliary column label.
    auxiliary_restored_column_name: basestring
        auxiliary column label.
    min_good_weeks: integer
        minimum number of 7-days sequences to be considered to
        extract the seasonal component.
    max_good_weeks: integer
        maximum number of 7-days sequences to be considered to
        extract the seasonal component; by default, it is set equal
        to `min_good_weeks` + 2.
    week_holiday_int: integer
        index corresponding to the week holiday, from 0 (Monday)
        to 6 (Sunday).
    nonnegative_constraint: boolean
        set to True to check and repair negative restored values.
    midweek_holidays_datetime_list: list
        the list of datetime related to midweek holidays

    Returns
    -------
    df: DataFrame
        reconstructed dataframe.
    """

    n_components_in_a_single_day = int(timedelta(days=1).total_seconds() / resolution.total_seconds())

    auxiliary_columns_to_drop = []
    auxiliary_date_column = '_nan_replacement_date_'
    auxiliary_time_column = '_nan_replacement_time_'
    auxiliary_columns_to_drop.append(auxiliary_date_column)
    auxiliary_columns_to_drop.append(auxiliary_time_column)

    df[auxiliary_date_column] = df[datetime_column].dt.date
    df[auxiliary_time_column] = df[datetime_column].apply(lambda x: get_daily_index(timestamp=x, resolution=resolution))

    midweek_date_list = list(set([x.date() for x in midweek_holidays_datetime_list]))
    partial_days_list = list(df[df[scan_column].isna()][auxiliary_date_column].unique())
    complete_days_list = [x for x in midweek_date_list if x not in partial_days_list]

    for curr_mid_day in partial_days_list:
        df_curr_mid_day = df[df[auxiliary_date_column] == curr_mid_day].copy()
        curr_datetime_to_be_replaced = list(df_curr_mid_day[df_curr_mid_day[scan_column].isna()][datetime_column])

        complete_sub_df = df[(df[auxiliary_date_column] >= curr_mid_day - timedelta(weeks=max_good_weeks)
                              ) & (df[auxiliary_date_column] <= curr_mid_day + timedelta(weeks=max_good_weeks)
                                   ) & (df[auxiliary_date_column].isin(complete_days_list))].copy()
        if len(complete_sub_df) == 0:
            complete_sub_df = df[(df[auxiliary_date_column] >= curr_mid_day - timedelta(weeks=min_good_weeks)
                                  ) & (df[auxiliary_date_column] <= curr_mid_day + timedelta(weeks=min_good_weeks)
                                       ) & (df[auxiliary_weekday_id_column] == week_holiday_int)].copy()

        decomposition_dict = get_daily_periodic_component(df=complete_sub_df, scan_column=scan_column,
                                                          date_column=auxiliary_date_column, resolution=resolution)

        if len(df_curr_mid_day[~ df_curr_mid_day[scan_column].isna()]) > 0.3 * n_components_in_a_single_day:
            trend_curr_mid_day = np.nanmean(df_curr_mid_day[scan_column].values - decomposition_dict['seasonal'])
        else:
            trend_curr_mid_day = decomposition_dict['avg_trend']

        df_curr_mid_day[scan_column] = df_curr_mid_day.apply(
            lambda x: trend_curr_mid_day + decomposition_dict['seasonal'][
                x[auxiliary_time_column]] if np.isnan(x[scan_column]) else x[scan_column], axis=1)

        # Check for any negative reconstructed components
        if nonnegative_constraint:
            df_curr_mid_day.loc[
                (df_curr_mid_day[scan_column] < 0.0) & (df_curr_mid_day[auxiliary_restored_column_name] == 1),
                scan_column] = 0.0

        for curr_timestamp in curr_datetime_to_be_replaced:
            df.loc[df[datetime_column] == curr_timestamp, scan_column] = df_curr_mid_day[
                df_curr_mid_day[datetime_column] == curr_timestamp][scan_column].iat[0]

    if len(auxiliary_columns_to_drop) > 0:
        df.drop(columns=auxiliary_columns_to_drop, inplace=True)

    return df
