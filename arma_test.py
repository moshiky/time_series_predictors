import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams


def main():
    rcParams['figure.figsize'] = 15, 6

    df_fx_data = pd.read_csv(r'datasets/BOE-XUDLERD.csv')
    df_fx_data['Date'] = pd.to_datetime(df_fx_data['Date'], format='%Y-%m-%d')
    indexed_df = df_fx_data.set_index('Date')

    ts = indexed_df['Value']
    # print(ts.head(5))

    # plt.plot(ts.index.to_pydatetime(), ts.values, color='red')
    # print(len(ts.values))
    # plt.show()

    # -- #1
    ts_week = ts.resample('W').mean()
    # print(len(ts_week.values))
    # plt.plot(ts_week.index.to_pydatetime(), ts_week.values, color='blue')
    # plt.show()

    # test_stationarity(ts_week)

    # -- #2
    ts_week_log = np.log(ts_week)
    # test_stationarity(ts_week_log)

    # -- #3
    decomposition = seasonal_decompose(ts_week)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    ts_week_log_select = ts_week_log[-80:]

    # plt.subplot(411)
    # plt.plot(ts_week_log_select.index.to_pydatetime(), ts_week_log_select.values, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(ts_week_log_select.index.to_pydatetime(), trend[-80:].values, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(ts_week_log_select.index.to_pydatetime(), seasonal[-80:].values, label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(ts_week_log_select.index.to_pydatetime(), residual[-80:].values, label='Residuals')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

    ts_week_log_diff = ts_week_log - ts_week_log.shift()
    # plt.plot(ts_week_log_diff.index.to_pydatetime(), ts_week_log_diff.values)
    # plt.show()

    ts_week_log_diff.dropna(inplace=True)
    # test_stationarity(ts_week_log_diff)

    # -- determine ARMA params using plots

    # ACF and PACF plots
    # lag_acf = acf(ts_week_log_diff, nlags=10)
    # lag_pacf = pacf(ts_week_log_diff, nlags=10, method='ols')
    #
    # # Plot ACF:
    # plt.subplot(121)
    # plt.plot(lag_acf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
    # plt.title('Autocorrelation Function')
    #
    # # Plot PACF:
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
    # plt.title('Partial Autocorrelation Function')
    # plt.tight_layout()
    # plt.show()

    # build ARMA model
    model = ARIMA(ts_week_log, order=(2, 1, 1))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(ts_week_log_diff.index.to_pydatetime(), ts_week_log_diff.values)
    plt.plot(ts_week_log_diff.index.to_pydatetime(), results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_week_log_diff) ** 2))
    plt.show()


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=52, center=False).mean()
    rolstd = timeseries.rolling(window=52, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue', label='Original')
    mean = plt.plot(rolmean.index.to_pydatetime(), rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.index.to_pydatetime(), rolstd.values, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


if __name__ == '__main__':
    main()
