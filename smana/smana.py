from smana.auxiliary_routines import *
from smana.utility import validate_input_params
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Union


def repair(input_df: pd.DataFrame, scan_column: str, datetime_column: str = None, trend_approx_days: int = 7,
           nonnegative_constraint: bool = False, holidays_stl: bool = False, week_holiday_int: int = 6,
           holidays_column: str = None, inplace: bool = False) -> Union[pd.DataFrame, None]:

    """Replace NaN values of a weekly-seasonal time series.

    This function restores missing values (NaN) of the time series
    `scan_column` in `input_df` dataframe, with `datetime_column` as
    datetime column, by a process based on the STL decomposition.
    Optionally, setting `holidays_stl` to True, it is possible to
    apply a similar strategy to repair missing data related to
    public holidays (this procedure is based on week holiday data).

    Parameters
    ----------
    input_df: DataFrame
        Input dataframe which collects the time series to be
        repaired, the datetime series and optionally the column
        with public holidays information.
    scan_column: string
        Label of the numeric column of `input_df` to be restored.
        Missing values must be represented as `numpy.NaN`.
    datetime_column: string
        Label of the datetime column of `input_df`; aware or naive
        datetime are supported. If unspecified, `input_df.index` is
        considered.
    trend_approx_days: integer, default 7
        Number of days to consider for trend estimation; higher
        values lead to approximations over longer periods. Integers
        less than 7 will be replaced by default value. It is not
        necessary to modify this parameter.
    nonnegative_constraint: boolean, default False
        Set to True to check and repair negative restored values.
    holidays_stl: boolean, default False
        Apply a specific strategy for the restoring of missing values
        related to public holidays.
    week_holiday_int: integer, default 6
        Index corresponding to the week holiday, from 0 (Monday)
        to 6 (Sunday). This argument is considered only if
        `holidays_stl` is set to True.
    holidays_column: string
        Label of the column which collects holidays information; for
        each row in `input_df`, the allowed values are only 0
        (working day) or 1 (holiday, including standard week
        holiday). This argument is considered only if `holidays_stl`
        is set to True.
    inplace : boolean, default False
        If False, return a copy. Otherwise, do operation
        inplace and return None.

    Returns
    -------
    input_df: DataFrame or None
        DataFrame restored or None if `inplace` is set to True.
    """

    lookup_weeks_shift = 1
    min_consecutive_good_weeks = 2
    max_consecutive_good_weeks = 3

    column_input_dt = 'starting_dt'
    column_current_dt = 'current_dt'
    column_input_values = 'starting_values'
    column_restored_flag = '_restored_'
    column_weekday_id = '_weekday_id_'
    column_holidays_flag = '_holidays_'

    validate_input_params(input_df, scan_column, datetime_column, trend_approx_days, nonnegative_constraint,
                          holidays_stl, week_holiday_int, holidays_column, inplace)

    if not inplace:
        input_df = input_df.copy(deep=True)

    if input_df[scan_column].isna().sum() == 0:
        if inplace:
            return None
        else:
            return input_df

    mapping_dt_df = pd.DataFrame()
    if datetime_column is None:
        mapping_dt_df[column_input_dt] = pd.Series(list(input_df.index))
    else:
        mapping_dt_df[column_input_dt] = pd.Series(list(input_df[datetime_column]))
    mapping_dt_df[column_input_values] = pd.Series(list(input_df[scan_column]))
    mapping_dt_df[column_current_dt] = mapping_dt_df[column_input_dt].apply(
        lambda x: x.replace(tzinfo=None))

    merging_columns_list = [column_current_dt, column_input_values]

    cur_holidays_date_list = list()
    if holidays_stl:
        merging_columns_list.append(column_holidays_flag)
        mapping_dt_df[column_holidays_flag] = pd.Series(list(input_df[holidays_column]))
        cur_holidays_date_list = list(
            mapping_dt_df[mapping_dt_df[column_holidays_flag] == 1][column_current_dt].dt.date.unique())

    # Handling any duplicates due to time changes
    mapping_dt_df.drop_duplicates(subset=column_current_dt, keep='first', inplace=True, ignore_index=True)

    # Infer data frequency
    inferred_resolution = min([
        mapping_dt_df[column_current_dt].iat[i + 1] - mapping_dt_df[column_current_dt].iat[i]
        for i in range(len(mapping_dt_df) - 1)])  # df time-resolution, as timedelta
    first_inner_dt = mapping_dt_df[column_current_dt].iat[0]
    last_inner_dt = mapping_dt_df[column_current_dt].iat[len(mapping_dt_df) - 1]

    # Inner datetime mask
    inner_df = pd.DataFrame()
    inner_periods = 1 + int((last_inner_dt - first_inner_dt).total_seconds() / inferred_resolution.total_seconds())
    inner_df[column_current_dt] = pd.date_range(start=first_inner_dt, end=last_inner_dt,
                                                periods=inner_periods, inclusive='both')

    inner_df = inner_df.merge(right=mapping_dt_df[merging_columns_list],
                              on=column_current_dt, how='left')

    midweek_holidays_reset = False
    midweek_holidays_datetime = list()
    midweek_holidays_df = pd.DataFrame()
    if holidays_stl:
        inner_df[column_weekday_id] = inner_df[column_current_dt].dt.weekday

        # Check if there are records with missing holidays flag
        if inner_df[column_holidays_flag].isna().sum() > 0:
            inner_df.loc[
                (inner_df[column_holidays_flag].isna()) & (
                    (inner_df[column_current_dt].dt.date.isin(cur_holidays_date_list)) | (
                        inner_df[column_weekday_id] == week_holiday_int
                    )),
                column_holidays_flag] = 1
            inner_df[column_holidays_flag] = inner_df[column_holidays_flag].fillna(value=0)

        midweek_holidays_df = inner_df[
            (inner_df[column_holidays_flag] == 1) & (inner_df[column_weekday_id] != week_holiday_int)].copy()
        midweek_holidays_datetime = list(midweek_holidays_df[column_current_dt])
        if len(midweek_holidays_df) > 0:
            midweek_holidays_reset = True
            inner_df.loc[inner_df[column_current_dt].isin(midweek_holidays_datetime), column_input_values] = np.nan

    trend_approx_size = int(timedelta(days=max(7, trend_approx_days)
                                      ).total_seconds() / inferred_resolution.total_seconds())

    inner_df[column_restored_flag] = pd.Series(0, dtype=int, index=inner_df.index)
    inner_df.loc[inner_df[column_input_values].isna(), column_restored_flag] = 1

    n_components_in_a_single_hour = int(timedelta(hours=1).total_seconds() / inferred_resolution.total_seconds())
    if n_components_in_a_single_hour == 0:
        max_nan_counter_single_holes = 1
    else:
        max_nan_counter_single_holes = n_components_in_a_single_hour

    is_df_restored = False
    main_df_nan = nan_presence_counter(df=inner_df, scan_column=column_input_values, datetime_column=column_current_dt,
                                       resolution=inferred_resolution)
    while not is_df_restored:

        if len(main_df_nan) == 0:
            is_df_restored = True
            continue

        evaluate_holes_distances(df_nan_list=main_df_nan, start_datetime=first_inner_dt,
                                 end_datetime=last_inner_dt, resolution=inferred_resolution)

        curr_isolated_holes_df_nan = main_df_nan[main_df_nan.long_side_weeks >= min_consecutive_good_weeks].copy()
        if len(curr_isolated_holes_df_nan) > 0:
            temp_max_symmetrical_weeks = curr_isolated_holes_df_nan.symmetrical_weeks.max()
            curr_isolated_holes_df_nan = curr_isolated_holes_df_nan[
                curr_isolated_holes_df_nan.symmetrical_weeks == temp_max_symmetrical_weeks].copy()
            max_first_dt = curr_isolated_holes_df_nan.first_dt.max()
            hole_info = curr_isolated_holes_df_nan[curr_isolated_holes_df_nan.first_dt == max_first_dt].copy()

            # Update the table of the remaining holes
            main_df_nan = main_df_nan[main_df_nan.first_dt != hole_info.first_dt.iat[0]].copy()

            curr_left_weeks = int(min(hole_info.left_weeks.iat[0], max_consecutive_good_weeks))
            curr_right_weeks = int(min(hole_info.right_weeks.iat[0], max_consecutive_good_weeks))

            curr_timedelta_left_shift = max(timedelta(weeks=curr_left_weeks), inferred_resolution * trend_approx_size)
            curr_timedelta_right_shift = max(timedelta(weeks=curr_right_weeks), inferred_resolution * trend_approx_size)

            sub_df = inner_df[
                (inner_df[column_current_dt] >= (hole_info.first_dt.iat[0] - curr_timedelta_left_shift)
                 ) & (inner_df[column_current_dt] <= (hole_info.last_dt.iat[0] + curr_timedelta_right_shift)
                      )].copy()

            sub_good_weeks_first_datetime = [
                hole_info.first_dt.iat[0] - timedelta(weeks=x) for x in range(curr_left_weeks, 0, -1)
                                            ] + [
                hole_info.last_dt.iat[0] + inferred_resolution + timedelta(weeks=y) for y in range(curr_right_weeks)
                                            ]

            seasonality = get_seasonal_component(df=sub_df, scan_column=column_input_values,
                                                 datetime_column=column_current_dt, resolution=inferred_resolution,
                                                 left_bound_list=sub_good_weeks_first_datetime)

            sub_df = substitute_holes(df=sub_df, scan_column=column_input_values, datetime_column=column_current_dt,
                                      resolution=inferred_resolution, df_nan_list=hole_info,
                                      seasonal_comp=seasonality, trend_approx_size=trend_approx_size)

            # Check for any negative reconstructed components
            if nonnegative_constraint:
                sub_df.loc[(sub_df[column_input_values] < 0.0
                            ) & (sub_df[column_restored_flag] == 1), column_input_values] = 0.0

            curr_datetime_to_be_replaced = list(sub_df[(sub_df[column_current_dt] >= hole_info.first_dt.iat[0]
                                                        ) & (sub_df[column_current_dt] <= hole_info.last_dt.iat[0])][
                                                    column_current_dt])
            for curr_timestamp in curr_datetime_to_be_replaced:
                inner_df.loc[inner_df[column_current_dt] == curr_timestamp, column_input_values
                             ] = sub_df[sub_df[column_current_dt] == curr_timestamp][column_input_values].iat[0]

            continue

        # Otherwise, interpolate short NaN-sequences
        df_1_holes = main_df_nan[main_df_nan.nan_counter <= max_nan_counter_single_holes].copy()
        if len(df_1_holes) > 0:
            df_best_1_holes = df_1_holes[df_1_holes.ratio == df_1_holes.ratio.max()].copy()
            if len(df_best_1_holes) > 1:
                df_best_1_holes = df_best_1_holes[df_best_1_holes.first_dt == df_best_1_holes.first_dt.max()].copy()

            # Update the table of the remaining holes
            main_df_nan = main_df_nan[main_df_nan.first_dt != df_best_1_holes.first_dt.iat[0]].copy()

            inner_df = interpolate_holes(df=inner_df, scan_column=column_input_values,
                                         datetime_column=column_current_dt, resolution=inferred_resolution,
                                         df_nan_list=df_best_1_holes,
                                         auxiliary_restored_column_name=column_restored_flag,
                                         nonnegative_constraint=nonnegative_constraint)
            continue

        df_2_holes = main_df_nan[main_df_nan.nan_counter <= 2 * max_nan_counter_single_holes].copy()
        if len(df_2_holes) > 0:
            df_best_2_holes = df_2_holes[df_2_holes.ratio == df_2_holes.ratio.max()].copy()
            if len(df_best_2_holes) > 1:
                df_best_2_holes = df_best_2_holes[df_best_2_holes.first_dt == df_best_2_holes.first_dt.max()].copy()

            # Update the table of the remaining holes
            main_df_nan = main_df_nan[main_df_nan.first_dt != df_best_2_holes.first_dt.iat[0]].copy()

            inner_df = interpolate_holes(df=inner_df, scan_column=column_input_values,
                                         datetime_column=column_current_dt, resolution=inferred_resolution,
                                         df_nan_list=df_best_2_holes,
                                         auxiliary_restored_column_name=column_restored_flag,
                                         nonnegative_constraint=nonnegative_constraint)
            continue

        # Last option, use lookup table
        df_nan_for_lookup_table = main_df_nan[main_df_nan.ratio == main_df_nan.ratio.max()].copy()
        if len(df_nan_for_lookup_table) > 1:
            df_nan_for_lookup_table = df_nan_for_lookup_table[
                df_nan_for_lookup_table.first_dt == df_nan_for_lookup_table.first_dt.max()].copy()

        # Update the table of the remaining holes
        main_df_nan = main_df_nan[main_df_nan.first_dt != df_nan_for_lookup_table.first_dt.iat[0]].copy()

        lookup_first_dt = df_nan_for_lookup_table.first_dt.iat[0]
        lookup_last_dt = df_nan_for_lookup_table.last_dt.iat[0]

        curr_valid_first_dt = max(first_inner_dt, lookup_first_dt - timedelta(weeks=lookup_weeks_shift))
        curr_valid_last_dt = min(last_inner_dt, lookup_last_dt + timedelta(weeks=lookup_weeks_shift))

        sub_df = inner_df[
            (inner_df[column_current_dt] >= curr_valid_first_dt) & (inner_df[column_current_dt] <= curr_valid_last_dt)
                          ].copy()
        sub_df['lookup_weekly_index'] = sub_df[column_current_dt].apply(
            lambda x: get_weekly_index(timestamp=x, resolution=inferred_resolution))
        new_values_sub_df = sub_df[(sub_df[column_current_dt] >= lookup_first_dt
                                    ) & (sub_df[column_current_dt] <= lookup_last_dt)].copy()
        new_values_sub_df[column_input_values] = new_values_sub_df.apply(
            lambda x: sub_df[sub_df.lookup_weekly_index == x.lookup_weekly_index
                             ][column_input_values].mean(skipna=True), axis=1)
        if new_values_sub_df[column_input_values].isna().sum() == len(new_values_sub_df):
            new_values_sub_df = sub_df[(sub_df[column_current_dt] >= lookup_first_dt - inferred_resolution
                                        ) & (sub_df[column_current_dt] <= lookup_last_dt + inferred_resolution)].copy()
        new_values_sub_df[column_input_values] = new_values_sub_df[column_input_values].interpolate(
            method='linear', limit_direction='both')
        # Check for any negative reconstructed components
        if nonnegative_constraint:
            new_values_sub_df.loc[
                (new_values_sub_df[column_input_values] < 0.0) & (new_values_sub_df[column_restored_flag] == 1),
                column_input_values] = 0.0

        curr_datetime_to_be_replaced = list(
            new_values_sub_df[(new_values_sub_df[column_current_dt] >= lookup_first_dt
                               ) & (new_values_sub_df[column_current_dt] <= lookup_last_dt)][column_current_dt])
        for curr_timestamp in curr_datetime_to_be_replaced:
            inner_df.loc[inner_df[column_current_dt] == curr_timestamp, column_input_values] = new_values_sub_df[
                new_values_sub_df[column_current_dt] == curr_timestamp][column_input_values].iat[0]

        continue

    holidays_cycle = False
    if midweek_holidays_reset:
        holidays_cycle = True
        # Restore nonstandard holidays records
        for curr_timestamp in midweek_holidays_datetime:
            inner_df.loc[inner_df[column_current_dt] == curr_timestamp, column_input_values] = midweek_holidays_df[
                midweek_holidays_df[column_current_dt] == curr_timestamp][column_input_values].iat[0]

    while holidays_cycle:

        holidays_df_nan = nan_presence_counter(df=inner_df, scan_column=column_input_values,
                                               datetime_column=column_current_dt, resolution=inferred_resolution)
        if len(holidays_df_nan) == 0:
            holidays_cycle = False
            continue

        df_holidays_2_holes = holidays_df_nan[holidays_df_nan.nan_counter <= 2 * max_nan_counter_single_holes].copy()
        if len(df_holidays_2_holes) > 0:
            inner_df = interpolate_holes(df=inner_df, scan_column=column_input_values,
                                         datetime_column=column_current_dt, resolution=inferred_resolution,
                                         df_nan_list=df_holidays_2_holes,
                                         auxiliary_restored_column_name=column_restored_flag,
                                         nonnegative_constraint=nonnegative_constraint)
            continue

        inner_df = midweek_holidays_replacement(df=inner_df, scan_column=column_input_values,
                                                datetime_column=column_current_dt, resolution=inferred_resolution,
                                                auxiliary_weekday_id_column=column_weekday_id,
                                                auxiliary_restored_column_name=column_restored_flag,
                                                week_holiday_int=week_holiday_int,
                                                nonnegative_constraint=nonnegative_constraint,
                                                min_good_weeks=min_consecutive_good_weeks,
                                                max_good_weeks=max_consecutive_good_weeks,
                                                midweek_holidays_datetime_list=midweek_holidays_datetime)

    # Update input_df
    mapping_dt_df = pd.merge(left=mapping_dt_df[[column_input_dt, column_current_dt]],
                             right=inner_df[[column_current_dt, column_input_values]],
                             on=column_current_dt, how='left')

    if datetime_column is None:
        replacement_indices_list = list(
            set(list(input_df[input_df[scan_column].isna()].index)
                ).intersection(
                set(list(mapping_dt_df[column_input_dt])))
        )
        for cur_dt in replacement_indices_list:
            input_df.loc[input_df.index == cur_dt, scan_column] = mapping_dt_df[
                mapping_dt_df[column_input_dt] == cur_dt][column_input_values].iat[0]

    else:
        replacement_dt_list = list(
            set(list(input_df[input_df[scan_column].isna()][datetime_column])
                ).intersection(
                set(list(mapping_dt_df[column_input_dt])))
        )
        for cur_dt in replacement_dt_list:
            input_df.loc[input_df[datetime_column] == cur_dt, scan_column] = mapping_dt_df[
                mapping_dt_df[column_input_dt] == cur_dt][column_input_values].iat[0]

    # Handle unrepaired records, due to clock changes
    if input_df[scan_column].isna().sum() > 0:
        input_df[scan_column] = input_df[scan_column].interpolate(method='linear', limit_direction='both')

    if inplace:
        return None
    else:
        return input_df
