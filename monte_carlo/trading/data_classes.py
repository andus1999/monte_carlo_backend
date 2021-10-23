from . import functions
import cbpro


def get_order_size(client: cbpro.AuthenticatedClient, trade_data_entry):
    if trade_data_entry['size'] is None:
        size = functions.get_filled_size(client, trade_data_entry['order_id'])
    else:
        size = float(trade_data_entry['size'])
    return size


class PendingInvestment(object):
    def __init__(self, ticker, limit_usd, forecast, prediction):
        self.ticker = ticker
        self.limit_usd = limit_usd
        self.size = None
        self.product_id = functions.get_product_id(ticker[0])
        self.buy_date = functions.get_current_timestamp()
        self.sell_date = self.buy_date + forecast * 3600 * 24
        self.prediction = prediction

    def setup_size(self, split_usd):
        self.size = split_usd / self.limit_usd

    def get_forecast(self):
        return (self.sell_date - self.buy_date) / 3600 / 24


class PendingDump(object):
    def __init__(self, client, trade_data_entry):
        self.order_id = trade_data_entry['order_id']
        self.ticker = trade_data_entry['ticker']
        if trade_data_entry['size'] is None:
            self.initial_size = get_order_size(client, trade_data_entry)
        else:
            self.initial_size = trade_data_entry['size']
        self.size = self.initial_size
        self.product_id = trade_data_entry['product_id']
        self.buy_date = trade_data_entry['buy_date']
        self.sell_date = trade_data_entry['sell_date']

    def postpone_sell_date(self, forecast):
        self.sell_date += forecast * 24 * 3600
