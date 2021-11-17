import json
import os
import datetime

from .. import settings
from monte_carlo.resources import filepaths, values
from monte_carlo.settings import models


def get_historical_data(coin):
    with open(os.path.join(os.path.dirname(__file__), filepaths.coins_json_path) + coin + '.json', 'r') as file:
        historical_data = json.load(file)
    return historical_data


def get_prediction_data(prediction_objects):
    model = prediction_objects[-1]['model']
    timestamp = datetime.datetime.combine(
        datetime.date.today()+datetime.timedelta(days=settings.forecast_in_days),
        values.min_time).timestamp()
    prediction_list = prediction_objects[-1]['prediction_list']
    prediction_data = {}
    for prediction_element in prediction_list:
        ticker = prediction_element['ticker']
        prediction = prediction_element['prediction']
        coin_id = ticker[4]
        current_metrics = get_historical_data(coin_id)[-1]
        prediction_data[coin_id] = {
            'name': ticker[1],
            'ticker': ticker[0],
            'id': coin_id,
            'market_data': {
                'open': current_metrics[0],
                'high': current_metrics[1],
                'low': current_metrics[2],
                'close': current_metrics[3],
                'volume': current_metrics[4],
                'market_cap': current_metrics[5],
                'timestamp': current_metrics[-1],
            },
            'prediction': {
                'low': prediction[1],
                'average': prediction[0],
                'high': prediction[2],
                'volatility': prediction_element['volatility'],
                'model_id': models.get_id(model),
                'timestamp': timestamp,
            },
        }
    return prediction_data


def get_model_data(prediction_objects):
    model = prediction_objects[-1]['model']
    time_series = list(map(lambda x: {'max_val_correlation': x['val_correlation'],
                                      'test_correlation': x['correlation'],
                                      'timestamp': x['date']}, prediction_objects))

    return {
        'model_id': models.get_id(model),
        'model_name': models.get_display_name(model),
        'correlation_data': time_series,
        'description': models.get_description(model),
    }
