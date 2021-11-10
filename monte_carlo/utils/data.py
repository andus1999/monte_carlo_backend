import json
import os
import datetime

from google.cloud.firestore_v1 import DELETE_FIELD

from ..resources import filepaths, values
from ..settings import models


def get_historical_data(coin):
    with open(os.path.join(os.path.dirname(__file__), filepaths.coins_json_path) + coin + '.json', 'r') as file:
        historical_data = json.load(file)
    return historical_data


def get_coin_data(prediction_element, model):
    ticker = prediction_element['ticker']
    prediction = prediction_element['prediction']
    volatility = prediction_element['volatility']
    forecast = prediction_element['forecast']
    date = datetime.date.today() + datetime.timedelta(days=forecast)
    dt = datetime.datetime.combine(date, values.min_time)
    timestamp = datetime.datetime.timestamp(dt)
    link = ticker[4]
    current_metrics = get_historical_data(link)[-1]
    return {
        'coin': ticker[1],
        'ticker': ticker[0],
        'id': link,
        'prediction': DELETE_FIELD,
        model: {
            'average': prediction[0],
            'low': prediction[1],
            'high': prediction[2],
            'volatility': volatility,
            'timestamp': timestamp
        },
        'market_data': {
            'timestamp': current_metrics[-1],
            'open': current_metrics[0],
            'high': current_metrics[1],
            'low': current_metrics[2],
            'close': current_metrics[3],
            'volume': current_metrics[4],
            'market_cap': current_metrics[5]
        }
    }


def get_model_data(prediction_objects):
    model = prediction_objects[-1]['model']
    max_val_correlations = list(map(lambda x: x['val_correlation'], prediction_objects))
    timestamps = list(map(lambda x: x['date'], prediction_objects))
    test_correlations = list(map(lambda x: x['correlation'], prediction_objects))
    print(test_correlations)
    return {
        'model_name': model,
        'correlation_graph': {
            'max_val_correlations': max_val_correlations,
            'test_correlations': test_correlations,
            'timestamps': timestamps,
        },
        'description': models.get_description(model),
    }
