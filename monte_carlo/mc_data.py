import csv
import datetime
import os
from os import path
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
# noinspection PyUnresolvedReferences
from tensorflow.math import divide_no_nan
from tensorflow.keras.utils import to_categorical
from . import settings
from .settings import hyper_parameters
import cbpro
scaler = MinMaxScaler()
warning_list = []
outlier_list = []


def get_coinbase_ticker_list():
    client = cbpro.PublicClient()
    products = client.get_products()
    coinbase_ticker_list = []
    for product in products:
        ticker = product['base_currency']
        if ticker not in coinbase_ticker_list:
            coinbase_ticker_list.append(ticker)
    return coinbase_ticker_list


def get_ticker_list():
    coinbase_ticker_list = get_coinbase_ticker_list()
    len_cb_list = len(coinbase_ticker_list)

    ticker_list = []

    with open(os.path.join(os.path.dirname(__file__), '../../data/crypto_names.csv'), newline='') as f:
        reader = filter(None, csv.reader(f))
        csv_data = list(reader)
        for i in range(0, len(csv_data)):
            ticker = csv_data[i][2]
            name = csv_data[i][1]
            index = int(csv_data[i][0])
            try:
                link = csv_data[i][3]
            except IndexError:
                link = None

            ticker_list.append([ticker, name, index, ticker in coinbase_ticker_list, link])
            if ticker in coinbase_ticker_list:
                coinbase_ticker_list.remove(ticker)

    cb = 0
    for i in range(0, len(ticker_list)):
        if ticker_list[i][3]:
            cb += 1

    print(ticker_list[0:5])
    print(len(ticker_list))
    print('\nCoinbase tickers:')
    print('Not found:')
    print(coinbase_ticker_list)
    print(str(cb) + '/' + str(len_cb_list) + ' found.')
    return ticker_list


def csv_to_pandas(ticker):
    if ticker[4] is not None:
        f_name = ticker[4]
    else:
        f_name = ticker[1].lower().replace(' ', '-').replace('.', '-')
    file_path = os.path.join(os.path.dirname(__file__), '../../cm/' + f_name + '.csv')
    if not path.exists(file_path):
        print('No file found: ' + str(f_name), end=' ')
        return pd.DataFrame()
    with open(file_path, newline='') as f:
        reader = filter(None, csv.reader(f))
        historical_data = list(reader)

        historical_data.reverse()
        num_col = len(historical_data[0])
        n_rows = len(historical_data)

        for r in range(0, n_rows):
            for c in range(0, num_col):
                value = historical_data[r][c]
                historical_data[r][c] = float(value)

        df = pd.DataFrame(historical_data, columns=['Open', 'High', 'Low', 'Close',
                                                    'Volume', 'Market Cap', 'Day',
                                                    'Month', 'Year', 'Ticker'])
        return df


def get_ticker_tables(ticker_list):
    tickers_for_data_unfiltered = []
    ticker_tables = []
    date = datetime.datetime.today()-datetime.timedelta(days=1)
    day = date.day
    month = date.month
    year = date.year
    not_up_to_date = []

    min_table_length = settings.forecast_in_days*2 + hyper_parameters.test_sample_size*2-1+settings.x_train_length
    for ticker in ticker_list:
        table = csv_to_pandas(ticker)
        if table.empty is False and len(table.index) >= min_table_length:
            last_entry = table.iloc[-1]
            if last_entry['Day'] == day and last_entry['Month'] == month and last_entry['Year'] == year:
                table.pop('Year')
                table.pop('Ticker')
                table.pop('Day')
                table.pop('Month')
                ticker_tables.append(table)
                tickers_for_data_unfiltered.append(ticker)
            else:
                not_up_to_date.append(ticker)

    print()
    if len(not_up_to_date) != 0:
        print('Tables not up to date:')
        print(not_up_to_date)
    else:
        print('All tables up to date.')
    print(len(ticker_tables))
    print(ticker_tables[0])
    return ticker_tables, tickers_for_data_unfiltered


def get_np_ticker_tables(ticker_tables, tickers_for_data_unfiltered, coinbase_only=False):
    np_ticker_tables = []
    tickers_for_data = []
    for i in range(0, len(tickers_for_data_unfiltered)):
        if coinbase_only is False or tickers_for_data_unfiltered[i][3] is True:
            np_table = ticker_tables[i].to_numpy().copy()
            np_ticker_tables.append(np.append([np.zeros_like(np_table[0])], np_table, axis=0))
            tickers_for_data.append(tickers_for_data_unfiltered[i])
    print('Tickers for data:')
    print(len(np_ticker_tables))

    volume_scaler = MinMaxScaler()
    market_cap_scaler = MinMaxScaler()

    volumes = []
    market_caps = []

    for np_ticker_table in np_ticker_tables:
        volumes.extend(np_ticker_table[:, 4].copy())
        market_caps.extend(np_ticker_table[:, 5].copy())

    scaled_volumes = volume_scaler.fit_transform(np.expand_dims(volumes, axis=1))
    scaled_market_caps = market_cap_scaler.fit_transform(np.expand_dims(market_caps, axis=1))

    i = 0
    for np_ticker_table in np_ticker_tables:
        for j in range(len(np_ticker_table)):
            np_ticker_table[j][4] = scaled_volumes[i]
            np_ticker_table[j][5] = scaled_market_caps[i]
            i += 1

    return np_ticker_tables, tickers_for_data


def get_data(tab, ticker, index):
    table = tab.copy()
    table_info = table[0:index + 1]
    price = table_info[:, 0:4]
    scaler.fit(price)

    table_slice = table[index - settings.x_train_length:index + 1]
    slice_price = table_slice[:, 0:4]
    slice_price_transformed = scaler.transform(slice_price)
    table_slice[:, 0:4] = slice_price_transformed
    x_data = table_slice

    if ticker is not None:
        ticker_id = ticker[2]
        m_data = to_categorical(ticker_id, hyper_parameters.number_of_tickers_for_data)
    else:
        m_data = None

    forecast = table[index + settings.forecast_in_days:index + settings.forecast_in_days + 1]
    if len(forecast) == 0:
        y_data = None
    else:
        slice_price = forecast[:, 0:4]
        slice_price_transformed = scaler.transform(slice_price)
        forecast[:, 0:4] = slice_price_transformed
        y_data = divide_no_nan(forecast[0][3], x_data[-1][3]).numpy()
        global warning_list
        global outlier_list
        if (0 in forecast or 0 in x_data[-1]) and ticker[1] not in warning_list and index > len(
                tab) - hyper_parameters.test_sample_size * 2 - settings.forecast_in_days * 2:
            warning_list.append(ticker[1])
        if (y_data > 3 or y_data < 0.1) and ticker[1] not in outlier_list and index > len(
                tab) - hyper_parameters.test_sample_size * 2 - settings.forecast_in_days * 2:
            outlier_list.append(ticker[1])

    return x_data, m_data, y_data


def generate_data(np_ticker_tables, tickers_for_data, generate_train_data, generate_validation_data=True):
    start_time = time.time()
    print('Forecast: ')
    print(str(settings.forecast_in_days) + ' days.')
    m_len = hyper_parameters.number_of_tickers_for_data
    columns = np_ticker_tables[0].shape[-1]
    train_len = 0
    test_len = 0
    val_len = 0
    for table in np_ticker_tables:
        for i in range(settings.x_train_length, len(table) - settings.forecast_in_days):
            min_test = len(table) - settings.forecast_in_days - hyper_parameters.test_sample_size
            max_val = min_test - settings.forecast_in_days
            min_val = max_val - hyper_parameters.test_sample_size + 1
            max_test = min_val - settings.forecast_in_days
            if min_test <= i:
                test_len += 1
            elif min_val <= i <= max_val and generate_validation_data:
                val_len += 1
            elif i <= max_test or (not generate_validation_data and i <= max_val):
                train_len += 1

    x_train = np.empty([train_len, settings.x_train_length + 1, columns], dtype=np.float32)
    m_train = np.empty([train_len, m_len], dtype=np.float32)
    y_train = np.empty([train_len], dtype=np.float32)
    x_test = np.empty([test_len, settings.x_train_length + 1, columns], dtype=np.float32)
    m_test = np.empty([test_len, m_len], dtype=np.float32)
    y_test = np.empty([test_len], dtype=np.float32)
    x_val = np.empty([val_len, settings.x_train_length + 1, columns], dtype=np.float32)
    m_val = np.empty([val_len, m_len], dtype=np.float32)
    y_val = np.empty([val_len], dtype=np.float32)

    test_index = 0
    train_index = 0
    val_index = 0

    for t in range(len(np_ticker_tables)):
        table = np_ticker_tables[t]
        ticker = tickers_for_data[t]

        if t % 100 == 0:
            print(str(t) + ' ' + str(ticker[1]), end=' ')
        for i in range(settings.x_train_length, len(table) - settings.forecast_in_days):
            min_test = len(table) - settings.forecast_in_days - hyper_parameters.test_sample_size
            max_val = min_test - settings.forecast_in_days
            min_val = max_val - hyper_parameters.test_sample_size + 1
            max_test = min_val - settings.forecast_in_days
            if min_test <= i:
                x_test[test_index], m_test[test_index], y_test[test_index] = get_data(table, ticker, i)
                test_index += 1
            elif min_val <= i <= max_val and generate_validation_data:
                x_val[val_index], m_val[val_index], y_val[val_index] = get_data(table, ticker, i)
                val_index += 1
            elif generate_train_data is False:
                pass
            elif i <= max_test or (not generate_validation_data and i <= max_val):
                x_train[train_index], m_train[train_index], y_train[train_index] = get_data(table, ticker, i)
                train_index += 1

    print("\nData warning zeros:")
    print(warning_list)
    print("\nData warning outlier:")
    print(outlier_list)

    # Outliers
    cut = 0.003
    low = np.quantile(y_train, cut)
    high = np.quantile(y_train, 1-cut)
    print('\nCut: '+str(cut))
    print('With outliers')
    print('Max: ' + str(np.max(y_train)))
    print('Min: ' + str(np.min(y_train)))
    print('Train samples: ' + str(len(x_train)))

    it = np.nditer(y_train, flags=['multi_index'])
    mask = np.ones(len(y_train), dtype=bool)
    for x in it:
        if x > high or x < low:
            mask[it.multi_index[0]] = False
    x_train = x_train[mask]
    m_train = m_train[mask]
    y_train = y_train[mask]

    print('\nNo outliers')
    print('Max: ' + str(np.max(y_train)))
    print('Min: ' + str(np.min(y_train)))
    print('Train samples: ' + str(len(x_train)))
    print('Validation samples: ' + str(len(x_val)))
    print('Test samples: ' + str(len(x_test)))
    print('\nTime: '+str(round(time.time()-start_time))+' s')
    if not generate_validation_data:
        x_val, m_val, y_val = None, None, None
    if not generate_train_data:
        x_train, m_train, y_train = None, None, None

    return x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test


def save_data(data_path, x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test):
    np.save(data_path + '/crypto_x_train', x_train)
    np.save(data_path + '/crypto_m_train', m_train)
    np.save(data_path + '/crypto_y_train', y_train)
    np.save(data_path + '/crypto_x_test', x_test)
    np.save(data_path + '/crypto_m_test', m_test)
    np.save(data_path + '/crypto_y_test', y_test)
    if x_val is not None and m_val is not None and y_val is not None:
        np.save(data_path + '/crypto_x_val', x_val)
        np.save(data_path + '/crypto_m_val', m_val)
        np.save(data_path + '/crypto_y_val', y_val)
    else:
        print('No validation data saved.')


def load_data(data_path, load_validation_data=True):
    x_train = np.load(data_path + '/crypto_x_train.npy')
    m_train = np.load(data_path + '/crypto_m_train.npy')
    y_train = np.load(data_path + '/crypto_y_train.npy')
    x_test = np.load(data_path + '/crypto_x_test.npy')
    m_test = np.load(data_path + '/crypto_m_test.npy')
    y_test = np.load(data_path + '/crypto_y_test.npy')
    if load_validation_data:
        x_val = np.load(data_path + '/crypto_x_val.npy')
        m_val = np.load(data_path + '/crypto_m_val.npy')
        y_val = np.load(data_path + '/crypto_y_val.npy')
        return x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test
    else:
        return x_train, m_train, y_train, None, None, None, x_test, m_test, y_test


def data(coinbase_only=False, generate_train_data=True):
    print('Generating data.')
    ticker_list = get_ticker_list()
    ticker_tables, tickers_for_data_unfiltered = get_ticker_tables(ticker_list)
    np_ticker_tables, tickers_for_data = get_np_ticker_tables(ticker_tables, tickers_for_data_unfiltered, coinbase_only)
    x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test = generate_data(np_ticker_tables,
                                                                                           tickers_for_data,
                                                                                           generate_train_data)
    return x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test, np_ticker_tables, tickers_for_data
