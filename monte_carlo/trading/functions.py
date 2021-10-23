import datetime
from typing import List

from . import setup, data_classes
from ..resources import values
from ..settings import hyper_parameters


def usd_to_quote(amount_usd, product_index):
    return amount_usd/setup.quote_currency_prices[product_index]


def usd_to_base(amount_usd, price_per_unit_usd):
    return amount_usd/price_per_unit_usd


def quote_to_usd(amount_quote, product_index):
    return amount_quote*setup.quote_currency_prices[product_index]


def base_to_usd(amount_base, price_per_unit_usd):
    return amount_base/price_per_unit_usd


def get_forecast(prediction_graph) -> int:
    dates = prediction_graph['dates']
    forecast = round(dates[-1] - dates[-2]) / 24 / 3600
    return int(forecast)


def investment_condition_satisfied(prediction_graph):
    investment_condition = hyper_parameters.investment_condition_factor*get_forecast(prediction_graph)
    currency_values = prediction_graph['predictions']
    prediction = currency_values[-1] / currency_values[-2] - 1
    if prediction > investment_condition:
        return True
    else:
        return False


def get_prediction_from_prediction_graph(prediction_graph):
    currency_values = prediction_graph['predictions']
    prediction = currency_values[-1] / currency_values[-2] - 1
    return prediction


def get_current_timestamp():
    current_datetime = datetime.datetime.combine(datetime.date.today(), values.min_time)
    current_timestamp = datetime.datetime.timestamp(current_datetime)
    return current_timestamp


def setup_investment_sizes(pending_investments: List[data_classes.PendingInvestment]):
    if len(pending_investments) == 0:
        return []
    investments = []
    forecast = pending_investments[0].get_forecast()
    for pending_investment in pending_investments:
        investment_condition = hyper_parameters.investment_condition_factor*forecast
        split_for_investment = hyper_parameters.investment / investment_condition * pending_investment.prediction / forecast
        investments.append(split_for_investment)
    capital = hyper_parameters.capital / forecast
    if sum(investments) > capital:
        factor = capital / sum(investments)
        investments = [n * factor for n in investments]
    for pending_investment, split_usd in zip(pending_investments, investments):
        pending_investment.setup_size(split_usd)
    return pending_investments


def get_product_id(base_currency):
    try:
        index = setup.base_currencies.index(base_currency)
        return setup.product_ids[index]
    except ValueError:
        return None


def update_trade_data_entry(trade_data, order_id, size, sell_date):
    for trade_data_entry in trade_data:
        if trade_data_entry['order_id'] == order_id:
            trade_data_entry['size'] = size
            trade_data_entry['sell_date'] = sell_date
    return trade_data


def remove_trade_data_entry(trade_data, order_id):
    trade_data = list(filter(lambda it: it['order_id'] != order_id, trade_data))
    return trade_data


def create_new_trade_data_entry(order_id, pending_investment: data_classes.PendingInvestment):
    trade_data_entry = {
        'order_id': order_id,
        'ticker': pending_investment.ticker,
        'size': None,
        'product_id': pending_investment.product_id,
        'buy_date': pending_investment.buy_date,
        'sell_date': pending_investment.sell_date
    }
    return trade_data_entry


def get_account_balance(currency):
    balance = None
    for account in setup.portfolio:
        if account['currency'] == currency:
            balance = account['balance']
    return float(balance)


def trade_history_entry_from_pending_order(pending_order, prediction_graphs, order_id):
    if isinstance(pending_order, data_classes.PendingInvestment):
        side = 'buy'
    elif isinstance(pending_order, data_classes.PendingDump):
        side = 'sell'
    else:
        raise ValueError('Input type {} not supported.'.format(type(pending_order)))
    current_market_price = None
    if get_current_market_price_usd(pending_order.ticker[0], prediction_graphs) is not None:
        current_market_price = \
            get_current_market_price_usd(pending_order.ticker[0], prediction_graphs)*pending_order.size
    trade_history_entry = {
        'order_id': order_id,
        'ticker': pending_order.ticker,
        'product_id': pending_order.product_id,
        'side': side,
        'size': pending_order.size,
        'value_usd': current_market_price,
        'date': datetime.datetime.now().timestamp()
    }
    return trade_history_entry


def get_current_market_price_usd(ticker, prediction_graphs):
    for prediction_graph in prediction_graphs:
        if prediction_graph['ticker'][0] == ticker:
            return prediction_graph['predictions'][-2]
    else:
        return None


def get_current_portfolio(portfolio_history, prediction_graphs):
    value_history = []
    for portfolio_history_entry in portfolio_history:
        value = 0
        for account in portfolio_history_entry['portfolio']:
            market_price_usd = get_current_market_price_usd(account['currency'], prediction_graphs)
            if market_price_usd is not None:
                value += market_price_usd*float(account['balance'])
            elif account['currency'] == 'USD':
                value += float(account['balance'])
            elif account['currency'] == 'EUR':
                value += float(account['balance'])*setup.euro_price_usd
        value_history.append({'value': value, 'date': datetime.datetime.now().timestamp()})

    return {'portfolio': portfolio_history[-1], 'value_history': value_history}


def get_filled_size(client, order_id):
    fills = client.get_fills(order_id=order_id)
    size = 0
    for fill in fills:
        size += float(fill['size'])
    return size


def float_to_string(value, decimal_places):
    return '{:.{}f}'.format(value, decimal_places)
