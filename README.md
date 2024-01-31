<div align="center">
  <img src="https://github.com/ToBe-Analytics/smana/blob/main/docs/images/tobeanalytics-positive-rgb.png?raw=true"><br>
</div>

-----------------

# smana: repairing tool for time series with weekly seasonality

## What is it?

**smana** is a Python package useful to restore missing values of a time series with a weekly pattern.

## Table of Contents

- [Main Features](#main-features)
- [Dependencies](#dependencies)
- [How it works](#how-it-works)
- [How to get it](#how-to-get-it)
- [Documentation](#documentation)
- [License](#license)
- [Contributing to smana](#contributing-to-smana)

## Main Features

  - Missing values restoring for time series with weekly seasonal pattern
  - Any time series with sub-daily resolution is supported
  - Handling of calendar information on public holidays (if provided by the user)

## Dependencies
- [statsmodels](https://www.statsmodels.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://www.numpy.org/)

## How it works
This package arises from the need to restore energy time series data, which usually present weekly seasonality and
not rarely even a correlation with public holidays. Nevertheless, the implementation is based only on the assumption 
that the time series shows a weekly pattern, thus this tool can be used to repair data of whatever nature with this 
seasonal characteristic.

The core of the algorithm is based on [STL decomposition](https://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf) 
("Seasonal and Trend decomposition using Loess"), a robust method for decomposing time series into trend, seasonal and 
remainder components, implemented in `statsmodels` module.

The main method of this package, `smana.repair()`, aims to restore sequences of missing data (represented as `numpy.NaN`
) by means of locally approximation of the trend and the seasonal components of the time series; in order to get the 
seasonality estimation, the algorithm tries to identify a sequence of at least 14 consecutive days of valid data: if 
it does not exist, linear interpolation or lookup table strategies are iteratively applied (using a ranking criteria 
on missing-values sequences) until a 14-days sequence appears.

In addition, this tool is able to handle calendar information on public holidays: this feature is useful only if the 
time series presents a correlation with these specific days, in particular if its daily pattern resemble that of 
standard week holidays; for this reason, it is recommended to leverage this feature only if this assumption is verified.

## How to get it

The source code is currently hosted on GitHub at:
https://github.com/ToBe-Analytics/smana

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/smana).

```sh
# PyPI
pip install -i https://pypi.org/simple/ smana
```

The list of changes to `smana` between each release can be found
[here](https://github.com/ToBe-Analytics/smana/releases). For full
details, see the commit logs at https://github.com/ToBe-Analytics/smana.

## Documentation

The package provides the following main method, which implements the whole procedure described:

**smana.repair**(***input_df***, ***scan_column***, ***datetime_column**=None*, ***trend_approx_days**=7*, 
                ***nonnegative_constraint**=False*, ***holidays_stl**=False*, ***week_holiday_int**=6*,
                ***holidays_column**=None*, ***inplace**=False*)

This function restores missing values (`numpy.NaN`) of the time series `scan_column` in `input_df` dataframe, 
with `datetime_column` as timestamps column, by a process based on the STL decomposition.
Optionally, setting `holidays_stl` to True, it is possible to apply a similar strategy to repair 
missing data related to public holidays (this procedure is based on week holiday data).

#### Parameters
* **input_df**: *pandas.DataFrame*  
Input dataframe which collects the time series to be repaired, the datetime series and optionally
the column with public holidays information.
* **scan_column**: *str*  
Label of the numeric column of *input_df* to be restored. Missing values must be represented as `numpy.NaN`.
* **datetime_column**: *str, default None*  
Label of the datetime column of *input_df*; aware or naive datetime are supported. If unspecified, 
*input_df.index* is considered.
* **trend_approx_days**: *int, default 7*  
Number of days to consider for trend estimation; higher values lead to approximations over longer periods.
Integers less than 7 will be replaced by default value. It is not necessary to modify this parameter.
* **nonnegative_constraint**: *bool, default False*  
Set to True to check and repair negative restored values.
* **holidays_stl**: *bool, default False*  
Apply a specific strategy for the restoring of missing values related to public holidays.
* **week_holiday_int**: *int, default 6*  
Index corresponding to the week holiday, from 0 (Monday) to 6 (Sunday). This argument is considered only if 
`holidays_stl` is set to True.
* **holidays_column**: *str, default None*  
Label of the column which collects holidays information; for each row in `input_df`, the allowed values are 
only 0 (working day) or 1 (holiday, including standard week holiday). This argument is considered only if 
`holidays_stl` is set to True.
* **inplace**:  *bool, default False*  
If False, return a copy. Otherwise, do operation inplace and the method returns None.

#### Returns
* **pandas.DataFrame or None**  
DataFrame restored or None if `inplace` is set to True.

## License
[BSD 3](LICENSE.txt)

## Contributing to smana
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
A detailed overview on how to contribute can be found in the [contributing guide](CONTRIBUTING.md).
As contributors and maintainers to this project, you are expected to abide by our code of conduct. 
More information can be found at: [Contributor Code of Conduct](CODE_OF_CONDUCT.md)

[Go to Top](#table-of-contents)
