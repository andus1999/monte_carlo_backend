import datetime
import json
import math
import os
import pickle
import time
from typing import List

from . import setup, functions
from ..resources import filepaths, strings
from .. import utils
from ..utils import logging
from . import data_classes
from ..settings import hyper_parameters, models
from .. import settings


def load_trade_files():
    path = os.path.join(os.path.dirname(__file__), filepaths.trade_data_path)
    trade_data = []
    trade_history = []
    portfolio_history = []
    if os.path.exists(path + filepaths.trade_data):
        with open(path + filepaths.trade_data, 'rb') as f:
            trade_data = pickle.load(f)
    if os.path.exists(path + filepaths.trade_history):
        with open(path + filepaths.trade_history) as f:
            trade_history = json.load(f)
    if os.path.exists(path + filepaths.portfolio_history):
        with open(path + filepaths.portfolio_history, 'rb') as f:
            portfolio_history = pickle.load(f)
    return trade_data, trade_history, portfolio_history


def save_trade_files(trade_data, trade_history, portfolio_history, current_portfolio):
    if settings.test_mode is True:
        print('Trading files are not saved in test mode.')
        return
    path = os.path.join(os.path.dirname(__file__), filepaths.trade_data_path)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + filepaths.trade_data, 'wb') as f:
        pickle.dump(trade_data, f, pickle.DEFAULT_PROTOCOL)
    with open(path + filepaths.trade_data_json, 'w+') as f:
        json.dump(utils.round_floats(trade_data), f)
    with open(path + filepaths.trade_history, 'w+') as f:
        json.dump(utils.round_floats(utils.trim_list(trade_history)), f)
    with open(path + filepaths.portfolio_history, 'wb') as f:
        pickle.dump(utils.trim_list(portfolio_history), f, pickle.DEFAULT_PROTOCOL)
    with open(path + filepaths.current_portfolio, 'w+') as f:
        json.dump(utils.round_floats(current_portfolio), f)
    print('Trade files updated.')


def load_prediction_graphs():
    path = os.path.join(os.path.dirname(__file__),
                        '../../../data/node_server/{}/prediction_graphs.json'.format(models.main_model))
    if os.path.exists(path):
        with open(path) as f:
            file = json.load(f)
        return file
    else:
        return []


def cancel_orders(client, trade_data, trade_history):
    canceled_orders = client.cancel_all()
    partially_filled_orders = []
    partially_filled_sizes = []
    new_trade_data = []
    for trade_data_entry in trade_data:
        order_id = trade_data_entry['order_id']
        if order_id not in canceled_orders:
            new_trade_data.append(trade_data_entry)
        else:
            filled_size = functions.get_filled_size(client, order_id)
            if filled_size != 0:
                partially_filled_orders.append(order_id)
                partially_filled_sizes.append(filled_size)
                print('Updating partially filled trade data entry from initial size {} to filled size {}'
                      .format(trade_data_entry['size'], filled_size))
                trade_data_entry['size'] = filled_size
                new_trade_data.append(trade_data_entry)
    for history_entry in trade_history:
        if history_entry['order_id'] in canceled_orders:
            if history_entry['order_id'] not in partially_filled_orders:
                print('Canceling {side} order of {size} USD. {value_usd}.'.format(side=history_entry['side'],
                                                                                  size=history_entry['size'],
                                                                                  value_usd=history_entry['value_usd']))
                history_entry['side'] = 'canceled'
            else:
                print('Updating history entry of partially filled order: {product_id} Value USD: {value_usd}.'
                      .format(value_usd=history_entry['value_usd'], product_id=history_entry['product_id']))
                index = partially_filled_orders.index(history_entry['order_id'])
                filled_size = partially_filled_sizes[index]
                size = history_entry['size']
                value_usd = history_entry['value_usd']
                history_entry['size'] = filled_size
                history_entry['value_usd'] = size / filled_size * value_usd
    return new_trade_data, trade_history


def buy(client, pending_investment: data_classes.PendingInvestment):
    time.sleep(1)
    mode = 'buy'
    base_currency = pending_investment.ticker[0]
    limit_usd = pending_investment.limit_usd
    volume_usd = pending_investment.size * limit_usd
    if base_currency not in setup.base_currencies:
        return None, None, None, '{} has no suitable trading pair.'.format(pending_investment.ticker[1])

    index = setup.base_currencies.index(base_currency)
    limit_quote = functions.usd_to_quote(limit_usd, index)
    limit_quote_str = functions.float_to_string(limit_quote, setup.quote_increments[index])

    base_min_size = setup.base_min_sizes[index]
    if pending_investment.size < base_min_size:
        print('Volume of {vol} too small. Using {min_vol} instead'.format(vol=pending_investment.size,
                                                                          min_vol=base_min_size))
        pending_investment.size = base_min_size
        volume_usd = functions.base_to_usd(pending_investment.size, limit_usd)
    volume_str = functions.float_to_string(pending_investment.size, setup.base_increments[index])

    trading_pair = setup.product_ids[index]

    print(mode + ' ' + trading_pair)
    print('Quote limit: ' + limit_quote_str)
    print('Size: ' + volume_str + ' / ' + str(volume_usd) + ' USD')

    out = client.place_limit_order(product_id=trading_pair,
                                   side=mode,
                                   price=limit_quote_str,  # QUOTE Units
                                   size=volume_str,  # BASE Units
                                   post_only=False)
    try:
        return out['id'], out['product_id'], pending_investment, None
    except KeyError:
        return None, None, None, ' API error: ' + str(out)


def sell(client, pending_dump: data_classes.PendingDump):
    time.sleep(1)
    mode = 'sell'
    base_currency = pending_dump.ticker[0]
    if base_currency not in setup.base_currencies:
        return None, None, None, '{} has no suitable trading pair.'.format(pending_dump.ticker[1])

    index = setup.base_currencies.index(base_currency)
    quote_currency_price = setup.quote_currency_prices[index]
    base_min_size = setup.base_min_sizes[index]
    if pending_dump.size < base_min_size:
        print('Volume of {vol} too small. Using {min_vol} instead'.format(vol=pending_dump.size,
                                                                          min_vol=base_min_size))
        pending_dump.size = base_min_size
    current_portfolio_balance = math.inf
    try:
        portfolio = client.get_accounts()
        current_portfolio_balance = \
            float(list(filter(lambda x: x['currency'] == base_currency, portfolio))[0]['available'])
    except IndexError:
        print('Could not get current balance of {}'.format(base_currency))
    if pending_dump.size > current_portfolio_balance:
        if current_portfolio_balance == 0:
            pending_dump.size = current_portfolio_balance
            print('Balance is 0. Removing trade data entry.')
            return None, None, pending_dump, None
        else:
            print('Volume of {vol} too large. Using current balance of {max_vol} instead.'
                  .format(vol=pending_dump.size, max_vol=current_portfolio_balance))
            pending_dump.size = current_portfolio_balance

    volume = pending_dump.size
    trading_pair = setup.product_ids[index]
    volume_str = functions.float_to_string(volume, setup.base_increments[index])
    price = float(client.get_product_ticker(trading_pair)['price'])

    print(mode + ' ' + trading_pair)
    print('Size: ' + volume_str + ' / ' + str(round(volume * price * quote_currency_price)) + ' USD')
    market_price = float(client.get_product_ticker(trading_pair)['price'])
    market_price_str = functions.float_to_string(market_price, setup.quote_increments[index])
    out = client.place_limit_order(product_id=trading_pair,
                                   side=mode,
                                   price=market_price_str,
                                   size=volume_str,
                                   post_only=False)
    try:
        return out['id'], out['product_id'], pending_dump, None
    except KeyError:
        return None, None, None, str(out)


def get_pending_investments(prediction_graphs):
    print('Getting pending investing.')
    print('Investment condition: {}'
          .format(hyper_parameters.investment_condition_factor * functions.get_forecast(prediction_graphs[0])))
    pending_investments = []
    for prediction_graph in prediction_graphs:
        if functions.investment_condition_satisfied(prediction_graph) is True and prediction_graph['ticker'][3] is True:
            ticker = prediction_graph['ticker']
            currency_values = prediction_graph['predictions']
            limit = currency_values[-2]
            pending_investments.append(
                data_classes.PendingInvestment(ticker, limit, functions.get_forecast(prediction_graph),
                                               functions.get_prediction_from_prediction_graph(prediction_graph)))
    pending_investments = functions.setup_investment_sizes(pending_investments)
    # disabled investments, should return pending_investments
    return pending_investments


def get_pending_dumps(client, trade_data):
    print('Getting pending dumps.')
    pending_dumps = []
    current_timestamp = functions.get_current_timestamp()
    for trade_data_entry in trade_data:
        if trade_data_entry['sell_date'] <= current_timestamp:
            pending_dumps.append(data_classes.PendingDump(client, trade_data_entry))
    return pending_dumps


def execute_trades(client,
                   pending_investments: List[data_classes.PendingInvestment],
                   pending_dumps: List[data_classes.PendingDump],
                   prediction_graphs,
                   trade_data,
                   trade_history,
                   portfolio_history):
    portfolio_history.append({'portfolio': setup.portfolio, 'date': datetime.datetime.now().timestamp()})
    if len(pending_dumps) == 0 and len(pending_investments) == 0:
        print('No pending dumps or pending investments to execute.')
        return trade_data, trade_history, portfolio_history
    for pending_dump in pending_dumps:
        for pending_investment in pending_investments:
            if pending_dump.product_id == pending_investment.product_id:
                investment_size = 0
                if pending_investment.size is not None:
                    investment_size = pending_investment.size
                difference = investment_size - pending_dump.initial_size
                print('{}:'.format(pending_investment.product_id))
                print('Pending investment and pending dump on the same asset.')
                print('Pending investment size: {}'.format(pending_investment.size))
                print('Pending dump size: {}'.format(pending_dump.initial_size))
                print('Difference: {}'.format(difference))
                if difference > 0:
                    print('Investment is greater than dump.')
                    print('Postponing the sell date of id {} and making additional investments.'.format(
                        pending_dump.order_id))
                    pending_dump.size = None
                    pending_dump.postpone_sell_date(functions.get_forecast(prediction_graphs[0]))
                    pending_investment.size = difference
                elif difference < 0:
                    print('Investment is smaller than dump.')
                    print('Selling less of id {} and postponing the remainder.'.format(pending_dump.order_id))
                    pending_dump.size -= investment_size
                    pending_dump.postpone_sell_date(functions.get_forecast(prediction_graphs[0]))
                    pending_investment.size = None
                else:
                    print('Investment and dump are the same.')
                    print('Postponing the sell date of id {}.'.format(pending_dump.order_id))
                    pending_dump.size = None
                    pending_dump.postpone_sell_date(functions.get_forecast(prediction_graphs[0]))
                    pending_investment.size = None
                print('Sizes adjusted to:')
                print('Pending investment size: {}'.format(pending_investment.size))
                print('Pending dump target: {}'.format(pending_dump.size))
    print('Dumping assets.')
    if len(pending_dumps) == 0:
        print('No assets to dump.')
    for pending_dump in pending_dumps:
        if pending_dump.size is None:
            trade_data = functions.update_trade_data_entry(trade_data,
                                                           pending_dump.order_id,
                                                           pending_dump.size,
                                                           pending_dump.sell_date)
            continue
        sell_order_id, product_id, pending_dump, error = sell(client, pending_dump)
        if error is None:
            print('Successfully sold.')
        else:
            print('Error: '+error)
        if pending_dump.size == pending_dump.initial_size or pending_dump.size == 0:
            trade_data = functions.remove_trade_data_entry(trade_data, pending_dump.order_id)
        else:
            trade_data = functions.update_trade_data_entry(trade_data,
                                                           pending_dump.order_id,
                                                           pending_dump.initial_size - pending_dump.size,
                                                           pending_dump.sell_date)
        trade_history.append(functions.trade_history_entry_from_pending_order(pending_dump,
                                                                              prediction_graphs,
                                                                              pending_dump.order_id))

    time.sleep(60)

    print('Investing in assets.')
    if len(pending_investments) == 0:
        print('No assets to invest in.')
    for pending_investment in pending_investments:
        if pending_investment.size is None:
            continue
        order_id, product_id, pending_investment, error = buy(client, pending_investment)
        if error is None:
            print('Successfully bought.')
            new_trade_data_entry = functions.create_new_trade_data_entry(order_id, pending_investment)
            trade_data.append(new_trade_data_entry)
            trade_history.append(functions.trade_history_entry_from_pending_order(pending_investment,
                                                                                  prediction_graphs,
                                                                                  order_id))
        else:
            print('Error: ' + error)
    print('All trades done.')
    return trade_data, trade_history, portfolio_history


def trade(prediction_graphs):
    logging.switch_logging_category(strings.logging_trading)
    client = setup.setup_client(prediction_graphs)
    trade_data, trade_history, portfolio_history = load_trade_files()
    trade_data, trade_history = cancel_orders(client, trade_data, trade_history)
    pending_investments = get_pending_investments(prediction_graphs)
    pending_dumps = get_pending_dumps(client, trade_data)
    trade_data, trade_history, portfolio_history = execute_trades(client,
                                                                  pending_investments,
                                                                  pending_dumps,
                                                                  prediction_graphs,
                                                                  trade_data,
                                                                  trade_history,
                                                                  portfolio_history)
    current_portfolio = functions.get_current_portfolio(portfolio_history, prediction_graphs)
    save_trade_files(trade_data, trade_history, portfolio_history, current_portfolio)
    logging.switch_to_previous_category()
