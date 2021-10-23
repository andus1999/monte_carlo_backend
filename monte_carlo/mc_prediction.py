import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import json
from . import mc_data
from . import settings
from .settings import hyper_parameters
from .resources import values, filepaths
from . import utils


def load_prediction_objects(model_name):
    path = os.path.join(os.path.dirname(__file__), filepaths.prediction_data_path + model_name + filepaths.prediction_objects_pkl)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            file = pickle.load(f)
        return file
    else:
        return []


def save_prediction_objects(prediction_objects, model_name):
    prediction_objects_trimmed = utils.trim_list(prediction_objects)
    path = os.path.join(os.path.dirname(__file__), filepaths.prediction_data_path + model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path+filepaths.prediction_objects_pkl, 'wb') as f:
        pickle.dump(prediction_objects_trimmed, f, pickle.DEFAULT_PROTOCOL)
    with open(path+filepaths.prediction_objects_json, 'w+') as f:
        json.dump(utils.round_floats(prediction_objects_trimmed), f)
    with open(path+filepaths.predictions_json, 'w+') as f:
        json.dump({'predictions': prediction_objects_trimmed[-1]['prediction_list']}, f)


def save_prediction_graphs(prediction_graphs, model_name):
    path = os.path.join(os.path.dirname(__file__), filepaths.prediction_data_path + model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path+filepaths.prediction_graphs_json, 'w+') as f:
        json.dump(utils.round_floats(prediction_graphs), f)


def extend_table(pred, table):
    new_data = np.multiply(np.expand_dims(table[-1], axis=0), pred)
    table_new = np.append(table, new_data, axis=0)

    return table_new


def save_div(a, b):
    if b == 0:
        return 0
    else:
        return a/b


def predict(dnn_model, ticker, tab, forks, disp_range):
    table = tab.copy()
    start_val = table[-1][3]
    x_data, m_data = mc_data.get_data(table, ticker, len(table) - 1)[0:2]
    x_data_arr = np.repeat(np.expand_dims(x_data, axis=0), forks, axis=0)
    m_data = np.repeat(np.expand_dims(m_data, axis=0), forks, axis=0)
    # forecast
    predictions = dnn_model.predict([x_data_arr, m_data])
    tables = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        tables.append(extend_table(prediction, table))
    tables = np.array(tables)

    predictions = np.mean(tables[:, -disp_range:, 3], axis=0)
    q1s = np.quantile(tables[:, -disp_range:, 3], 0.01, axis=0)
    q3s = np.quantile(tables[:, -disp_range:, 3], 0.99, axis=0)
    gain = save_div(predictions[-1], start_val) - 1
    quant1 = save_div(q1s[-1], start_val) - 1
    quant3 = save_div(q3s[-1], start_val) - 1

    return gain, quant1, quant3, predictions, q1s, q3s


def get_predictions(dnn_model, ticker, table, forks):
    today = datetime.date.today()
    disp_range = 20
    gain, quant1, quant3, predictions, q1s, q3s = predict(dnn_model, ticker=ticker, tab=table,
                                                          forks=forks, disp_range=disp_range)
    dates = []
    for j in range(disp_range-2, -1, -1):
        dates.append(today - datetime.timedelta(days=j))
    dates.append(today + datetime.timedelta(days=settings.forecast_in_days))

    return gain, quant1, quant3, predictions, q1s, q3s, dates


def show_predictions(dnn_model, np_ticker_tables, tickers_for_data):
    tickers = len(tickers_for_data)
    print('Prediction forks: ' + str(hyper_parameters.prediction_forks))
    gains = []
    quant1s = []
    quant3s = []
    for i in range(tickers):
        table = np_ticker_tables[i]
        ticker = tickers_for_data[i]
        gain, quant1, quant3, predictions, q1s, q3s, dates = get_predictions(dnn_model, ticker, table, hyper_parameters.prediction_forks)
        print(gain)
        plt.plot_date(dates, predictions, linestyle='-')
        plt.fill_between(dates, q1s, q3s, alpha=0.2)
        plt.xticks([dates[0], dates[len(dates)//2], dates[-1]])
        plt.title(ticker[1])
        plt.show()
        plt.close()
        gains.append(gain)
        quant1s.append(quant1)
        quant3s.append(quant3)

    winner_gain = np.quantile(gains, 0.9)

    for i in range(tickers):
        if gains[i] > winner_gain:
            print(tickers_for_data[i][1])
            print(round(gains[i], 2))
            print(round(quant1s[i], 2))
            print(round(quant3s[i], 2))


def update_files(prediction_object, prediction_graphs, model_name='default'):
    save_prediction_graphs(prediction_graphs, model_name)

    prediction_objects = load_prediction_objects(model_name)
    for old_prediction_object in prediction_objects:
        old_prediction_object['prediction_list'] = []
    prediction_objects.append(prediction_object)
    save_prediction_objects(prediction_objects, model_name)
    print('Files updated.')


def get_prediction_list_and_graphs(dnn_model, model_name, model_metrics, np_ticker_tables, tickers_for_data):
    print('Prediction forks: '+str(hyper_parameters.prediction_forks))
    print('Prediction forecast: '+str(settings.forecast_in_days))
    model_performance = model_metrics[0]
    investing_performance = model_metrics[1]
    correlation = model_metrics[2]
    val_correlation = model_metrics[3]
    prediction_graphs = []
    prediction_list = []
    now = datetime.datetime.now().timestamp()
    prediction_spreads = []
    for i in range(len(tickers_for_data)):
        ticker = tickers_for_data[i]
        table = np_ticker_tables[i]
        gain, quant1, quant3, predictions, q1s, q3s, dates = get_predictions(dnn_model, ticker, table, hyper_parameters.prediction_forks)
        timestamps = []
        for date in dates:
            dt = datetime.datetime.combine(date, values.min_time)
            timestamp = datetime.datetime.timestamp(dt)
            timestamps.append(timestamp)
        prediction_graph = {'ticker': ticker, 'predictions': predictions.tolist(), 'low': q1s.tolist(), 'high': q3s.tolist(), 'dates': timestamps}
        prediction_graphs.append(prediction_graph)
        prediction = {'ticker': ticker, 'prediction': [gain, quant1, quant3]}
        prediction_list.append(prediction)
        prediction_spreads.append(quant3 - quant1)
    min_spread = min(prediction_spreads)
    prediction_spreads -= min_spread
    max_spread = max(prediction_spreads)
    prediction_spreads /= max_spread
    for prediction, volatility in zip(prediction_list, prediction_spreads):
        prediction['volatility'] = volatility
    prediction_object = {'prediction_list': prediction_list, 'model': model_name, 'performance': model_performance,
                         'investing_performance': investing_performance,
                         'correlation': correlation, 'val_correlation': val_correlation, 'date': now}
    return prediction_object, prediction_graphs


def predict_and_update_files(dnn_model, model_name, model_metrics, np_ticker_tables, tickers_for_data):
    prediction_object, prediction_graphs = get_prediction_list_and_graphs(dnn_model, model_name, model_metrics, np_ticker_tables, tickers_for_data)
    update_files(prediction_object, prediction_graphs, model_name)
    return prediction_graphs
