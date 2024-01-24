import pandas as pd
from typing import Union


def _raise_type_error(var_name: str, allowed_types: Union[list, str]) -> None:

    if isinstance(allowed_types, str):
        full_types_str = allowed_types
    elif len(allowed_types) == 1:
        full_types_str = allowed_types[0]
    else:
        full_types_str = ', '.join(allowed_types[:-1]) + f' or {allowed_types[-1]}'

    raise TypeError(f'{var_name} must be a {full_types_str}')


def _raise_value_error(var_name: str, allowed_values_string: str) -> None:
    raise ValueError(f'Allowed values for {var_name}: {allowed_values_string}')


def validate_input_params(input_df, scan_column, datetime_column, trend_approx_days, nonnegative_constraint,
                          holidays_stl, week_holiday_int, holidays_column, inplace) -> None:

    if not isinstance(input_df, pd.DataFrame):
        _raise_type_error('input_df', 'dataframe')

    if not isinstance(scan_column, str):
        _raise_type_error('scan_column', 'string')
    if scan_column not in input_df.columns:
        raise AttributeError(f"input_df has no column '{scan_column}' (indicated as scan_column)")

    if not isinstance(datetime_column, (str, type(None))):
        _raise_type_error('datetime_column', ['string', 'None'])
    if isinstance(datetime_column, str):
        if datetime_column not in input_df.columns:
            raise AttributeError(f"input_df has no column '{datetime_column}' (indicated as datetime_column)")

    if not isinstance(trend_approx_days, int):
        _raise_type_error('trend_approx_days', 'integer')

    if not isinstance(nonnegative_constraint, bool):
        _raise_type_error('nonnegative_constraint', 'boolean')

    if not isinstance(holidays_stl, bool):
        _raise_type_error('holidays_stl', 'boolean')

    if not isinstance(week_holiday_int, int):
        _raise_type_error('week_holiday_int', 'integer')
    if (week_holiday_int < 0) or (week_holiday_int > 6):
        _raise_value_error('week_holiday_int', 'from 0 (Monday) to 6 (Sunday)')

    if not isinstance(holidays_column, (str, type(None))):
        _raise_type_error('holidays_column', ['string', 'None'])

    if holidays_stl:
        if holidays_column not in input_df.columns:
            raise AttributeError(f"input_df has no column '{holidays_column}' (indicated as holidays_column)")

        if len(set(list(input_df[holidays_column].unique())).difference({0, 1})) > 0:
            _raise_value_error(f"input_df column '{holidays_column}' (indicated as holidays_column)",
                               "0 (working day) and 1 (holiday)")

    if not isinstance(inplace, bool):
        _raise_type_error('inplace', 'boolean')

    return
