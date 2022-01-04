import csv
import os
import json
from typing import Optional, List
import datetime

import binance.exceptions
import cbpro
import requests.exceptions

from .utils import functions
from .resources import filepaths

from binance import Client


def load_historical_data(name):
    with open(os.path.join(os.path.dirname(__file__), filepaths.binance_folder+name+'.json'), 'r') as f:
        historical_data = json.load(f)
    return historical_data


def save_historical_data(name, historical_data):
    with open(os.path.join(os.path.dirname(__file__), filepaths.binance_folder+name+'.json'), 'w') as f:
        json.dump(historical_data, f)


def get_coinbase_ticker_list():
    cb_client = cbpro.PublicClient()
    products = cb_client.get_products()
    global coinbase_ticker_list
    coinbase_ticker_list = []
    for product in products:
        ticker = product['base_currency']
        if ticker not in coinbase_ticker_list:
            coinbase_ticker_list.append(ticker)


def create_client():
    binance_client = Client()
    with open(os.path.join(os.path.dirname(__file__), filepaths.binance_crypto_names_csv), newline='') as f:
        reader = filter(None, csv.reader(f))
        global crypto_list
        crypto_list = list(reader)
    get_coinbase_ticker_list()
    return binance_client


def get_link(index):
    if crypto_list is None:
        raise ValueError('Client not initialized.')
    try:
        name = crypto_list[index][3]
    except IndexError:
        name = crypto_list[index][1].lower().replace(' ', '-').replace('.', '-')
    return name


def get_ticker(index):
    if crypto_list is None:
        raise ValueError('Client not initialized.')
    ticker = crypto_list[index][2]
    return ticker


def get_historical_data(index, start, end):
    ticker = get_ticker(index)
    try:
        historical_data = client.get_historical_klines(ticker+"USDT", Client.KLINE_INTERVAL_1DAY, start, end)
        return historical_data
    except (binance.exceptions.BinanceAPIException, requests.exceptions.ConnectionError) as e:
        if e is binance.exceptions.BinanceAPIException:
            print('No USDT-pair found on binance. {} {}'.format(get_link(index), get_ticker(index) in coinbase_ticker_list))
            return None
        if e is requests.exceptions.ConnectionError:
            print('Requests Connection Error.')


def get_table_for_coinmarketcap(table, index, update_only):
    if update_only is True:
        timedelta = datetime.timedelta(days=30)
    else:
        timedelta = datetime.timedelta(days=365*10)
    start = (functions.get_yesterday_datetime()-timedelta).strftime("%d %b, %Y")
    end = datetime.datetime(table[0][-2], table[0][-3], table[0][-4], 0, 0, 0, 0, datetime.timezone.utc).strftime("%d %b, %Y")
    historical_data = None
    try:
        historical_data = get_historical_data(index, start, end)
    except requests.exceptions.ReadTimeout:
        pass
    if historical_data is None or len(historical_data) == 0:
        # print(f'Binance data invalid. {get_link(index)}')
        return table
    historical_data.reverse()
    for i in range(min(len(historical_data), len(table))):
        table[i][0] = float(historical_data[i][1])
        table[i][1] = float(historical_data[i][2])
        table[i][2] = float(historical_data[i][3])
        table[i][3] = float(historical_data[i][4])
    return table


def add_coin(index):
    start = datetime.datetime(2010, 1, 1, 0, 0, 0, 0, datetime.timezone.utc).strftime("%d %b, %Y")
    end = functions.get_yesterday_datetime().strftime("%d %b, %Y")
    historical_data = get_historical_data(index, start, end)
    if historical_data is not None:
        name = get_link(index)
        save_historical_data(name, historical_data)


def update_coin(index):
    start = (functions.get_yesterday_datetime()-datetime.timedelta(days=30)).strftime("%d %b, %Y")
    end = functions.get_yesterday_datetime().strftime("%d %b, %Y")
    new_historical_data = get_historical_data(index, start, end)
    name = get_link(index)
    historical_data = load_historical_data(name)
    for i in range(len(new_historical_data)):
        if new_historical_data[i][0] > historical_data[-1][0]:
            historical_data.append(new_historical_data[i])
    save_historical_data(name, historical_data)


def add_coins(indices):
    for index in indices:
        add_coin(index)


def update_coins(indices):
    for index in indices:
        update_coin(index)


crypto_list: Optional[List] = None
coinbase_ticker_list: Optional[List] = None
client = create_client()

