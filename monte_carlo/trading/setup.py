from typing import Optional

from .. import settings
from ..resources import filepaths
from . import functions
import os
import json
import cbpro


quote_currencies: Optional[list]


def get_authentication_data(key_file_name):
    firebase_tokens_path = os.path.join(os.path.dirname(__file__), filepaths.api_key_path + key_file_name)
    with open(firebase_tokens_path) as f:
        data = json.load(f)
        api_key = data['api_key']
        passphrase = data['passphrase']
        secret = data['secret']
    return api_key, secret, passphrase


def get_sandbox_client():
    key, b64secret, passphrase = get_authentication_data(filepaths.sandbox_key)
    return cbpro.AuthenticatedClient(key, b64secret, passphrase,
                                     api_url="https://api-public.sandbox.pro.coinbase.com")


def get_test_client():
    key, b64secret, passphrase = get_authentication_data(filepaths.test_key)
    return cbpro.AuthenticatedClient(key, b64secret, passphrase)


def setup_trading_pairs(client):
    products = client.get_products()
    available_quote_currencies = ['EUR', 'USDC', 'DAI', 'BTC', 'USDT']
    global product_ids
    product_ids = []
    global base_currencies
    base_currencies = []
    global quote_currencies
    quote_currencies = []
    global base_increments
    base_increments = []
    global quote_increments
    quote_increments = []
    global base_min_sizes
    base_min_sizes = []

    for product in products:
        quote_currency = product['quote_currency']
        if quote_currency in available_quote_currencies:
            base_currency = product['base_currency']
            product_id = product['id']
            base_increment = len(product['base_increment'])-2
            if base_increment < 0:
                base_increment = 0
            quote_increment = len(product['quote_increment'])-2
            if quote_increment < 0:
                quote_increment = 0
            base_min_size = float(product['base_min_size'])
            if base_currency not in base_currencies:
                product_ids.append(product_id)
                base_currencies.append(base_currency)
                quote_currencies.append(quote_currency)
                base_increments.append(base_increment)
                quote_increments.append(quote_increment)
                base_min_sizes.append(base_min_size)
            else:
                index_of_pair = base_currencies.index(base_currency)
                old_quote_currency = quote_currencies[index_of_pair]
                if available_quote_currencies.index(quote_currency) < available_quote_currencies.index(old_quote_currency):
                    product_ids[index_of_pair] = product_id
                    quote_currencies[index_of_pair] = quote_currency
                    base_increments[index_of_pair] = base_increment
                    quote_increments[index_of_pair] = quote_increment
                    base_min_sizes[index_of_pair] = base_min_size
    return products, base_currencies, quote_currencies


def setup_quote_currency_prices(client, prediction_graphs):
    bitcoin_price_usd = functions.get_current_market_price_usd('BTC', prediction_graphs)
    global euro_price_usd
    try:
        usd_price_euro = float(client.get_product_ticker('USDT-EUR')['price'])
    except KeyError:
        usd_price_euro = 1.2
        print('Assuming USD-EUR to be {}'.format(usd_price_euro))
    euro_price_usd = 1/usd_price_euro

    global quote_currency_prices
    quote_currency_prices = []
    if quote_currencies is None:
        raise AssertionError('Trading pairs need to be set up before calling {}'
                             .format(setup_quote_currency_prices.__name__))
    for quote_currency in quote_currencies:
        if quote_currency == 'EUR':
            quote_currency_prices.append(euro_price_usd)
        elif quote_currency == 'BTC':
            quote_currency_prices.append(bitcoin_price_usd)
        else:
            quote_currency_prices.append(1)


def get_account_balances(client):
    global portfolio
    portfolio = client.get_accounts()


def setup_sandbox_client(prediction_graphs):
    print('\nInitializing sandbox client.')
    client = get_sandbox_client()
    setup_trading_pairs(client)
    setup_quote_currency_prices(client, prediction_graphs)
    get_account_balances(client)
    return client


def setup_test_client(prediction_graphs):
    print('\nInitializing test client.')
    client = get_test_client()
    setup_trading_pairs(client)
    setup_quote_currency_prices(client, prediction_graphs)
    get_account_balances(client)
    return client


def setup_client(prediction_graphs):
    if settings.test_mode is False:
        return setup_test_client(prediction_graphs)
    else:
        return setup_sandbox_client(prediction_graphs)


product_ids, base_currencies, quote_currencies = None, None, None
base_increments, quote_increments, base_min_sizes = None, None, None
quote_currency_prices = None
portfolio = None
euro_price_usd = None
