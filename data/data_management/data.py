import csv
import datetime
import os

import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

from ..resources import filepaths


def get_coin_data_json(data, name, ticker, coin_id):
    global meta_data
    if meta_data is None:
        meta_data = {}
        initialize()
    json_data = []
    for date in data:
        json_data.append({
            'open': date[0],
            'high': date[1],
            'low': date[2],
            'close': date[3],
            'volume': date[4],
            'market_cap': date[5],
            'timestamp': date[6],
        })
    description = meta_data[coin_id]['description']
    logo = meta_data[coin_id]['logo']
    website = first_or_none(meta_data[coin_id]['urls']['website'])
    technical_doc = first_or_none(meta_data[coin_id]['urls']['technical_doc'])
    source_code = first_or_none(meta_data[coin_id]['urls']['source_code'])

    sentiment_data = load_sentiment(coin_id)
    sentiment = sentiment_data['sentiment']
    headlines = sentiment_data['headlines']

    return {
            'technical_doc': technical_doc,
            'source_code': source_code,
            'website': website,
            'logo': logo,
            'description': description,
            'historical_data': json_data,
            'name': name,
            'ticker': ticker,
            'coin_id': coin_id,
            'headlines': headlines,
            'sentiment': sentiment,
            'timestamp': timestamp,
        }


def initialize():
    global meta_data
    global timestamp
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/info'
    crypto_list = load_csv()
    slugs = ''
    for i in range(0, len(crypto_list)):
        slugs += get_link(i, crypto_list) + ','
    parameters = {
        'slug': slugs[:-1],
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '5b2b95fa-2f15-47f1-9563-b144c73c16cc',
    }
    try:
        response = requests.get(url, headers=headers, params=parameters)
        data = json.loads(response.text)
        id_data = data['data']
        meta_data = {}
        for id_key in id_data.keys():
            obj = id_data[id_key]
            meta_data[obj['slug']] = obj

    except (ConnectionError, Timeout, TooManyRedirects) as e:
        meta_data = None
        print(e)


def load_csv():
    with open(os.path.join(os.path.dirname(__file__), '../'+filepaths.crypto_names_csv), newline='') as f:
        reader = filter(None, csv.reader(f))
        crypto_list = list(reader)
    return crypto_list


def load_sentiment(coin_id):
    path = os.path.join(os.path.dirname(__file__), filepaths.sentiment_path) + coin_id + '.json'
    if os.path.exists(path):
        with open(path) as f:
            sentiment = json.load(f)[-1]
    else:
        sentiment = {
            'sentiment': {
                'sentiment_score': None,
                'sentiment_value': None,
            },
            'headlines': [],
            'timestamp': datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
        }
    return sentiment


def first_or_none(data):
    try:
        return data[0]
    except IndexError:
        return None


def get_link(index, crypto_list):
    try:
        name = crypto_list[index][3]
    except IndexError:
        name = crypto_list[index][1].lower().replace(' ', '-').replace('.', '-')
    return name


meta_data = None
timestamp = None
initialize()
