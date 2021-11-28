import datetime
import json
import os
import numpy as np
from . import mc_prediction, mc_evaluation, mc_model, mc_data, trading
from .settings import hyper_parameters, models
from . import settings, utils
from .utils import messaging, logging
from .resources import strings, filepaths


def get_best_model_name(model_list):
    best_performance = -np.Inf
    best_model = None
    for model in model_list:
        if model['metrics'][1] > best_performance:
            best_performance = model['metrics'][1]
            best_model = model
    best_model_name = best_model['name']
    print('Best performing model on train set: ' + best_model_name)
    return best_model_name


def save_model_list(model_data_list):
    path = os.path.join(os.path.dirname(__file__), filepaths.model_data_path)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + filepaths.model_data, 'w+') as f:
        json.dump(utils.round_floats(model_data_list), f)


def create_log_dir(name):
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    model_log_path = os.path.join(os.path.dirname(__file__),
                                  '../../cm_model/logs/' + name + '/')
    if not os.path.exists(model_log_path):
        os.mkdir(model_log_path)
    log_path = os.path.join(os.path.dirname(__file__),
                            '../../cm_model/logs/' + name + '/' + time_string + '/')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return log_path


def predict_and_retrain(model_name, epochs):
    settings.init_settings(model_name)
    log_path = create_log_dir(model_name)

    x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test, np_ticker_tables, tickers_for_data = mc_data.data()

    dnn_model = mc_model.load_model(model_name)
    model_metrics = mc_evaluation.evaluate_model(dnn_model, model_name, tickers_for_data, x_train, m_train, y_train, x_val, m_val,
                                                 y_val, x_test, m_test,
                                                 y_test, log_path)

    mc_prediction.predict_and_update_files(dnn_model, model_name, model_metrics, np_ticker_tables, tickers_for_data)
    if epochs > 0:
        mc_model.retrain_model(x_train, m_train, y_train, x_val, m_val, y_val, model_name=model_name, epochs=epochs)


def evaluate(model_name):
    settings.init_settings(model_name)
    log_path = create_log_dir(model_name)

    x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test, np_ticker_tables, tickers_for_data = mc_data.data(
        generate_train_data=False)

    dnn_model = mc_model.load_model(model_name)
    mc_evaluation.evaluate_model(dnn_model, model_name, tickers_for_data, x_train, m_train, y_train, x_val, m_val, y_val, x_test,
                                 m_test,
                                 y_test, log_path)


def retrain(model_name, epochs):
    settings.init_settings(model_name)
    x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test, np_ticker_tables, tickers_for_data = mc_data.data()

    if epochs > 0:
        mc_model.retrain_model(x_train, m_train, y_train, x_val, m_val, y_val, model_name=model_name, epochs=epochs)


def update_predictions(model_name):
    settings.init_settings(model_name)
    log_path = create_log_dir(model_name)
    x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, y_test, np_ticker_tables, tickers_for_data = mc_data.data(
        generate_train_data=False)

    dnn_model = mc_model.load_model(model_name)
    model_metrics = mc_evaluation.evaluate_model(dnn_model, model_name, tickers_for_data, x_train, m_train, y_train, x_val, m_val,
                                                 y_val, x_test, m_test,
                                                 y_test, log_path)

    prediction_graphs = mc_prediction.predict_and_update_files(dnn_model, model_name, model_metrics, np_ticker_tables, tickers_for_data)
    return model_metrics, prediction_graphs


def multi_model_predict():
    logging.switch_logging_category(strings.logging_multi_model_predict)
    model_list = []
    for model_name in models.lstm_models + models.dense_models:
        model_metrics, prediction_graphs = update_predictions(model_name)
        if model_name == models.main_model:
            trading.trade(prediction_graphs)
        model_list.append({'name': model_name, 'description': models.get_description(model_name),
                           'metrics': model_metrics, 'forecast': settings.forecast_in_days})
    model_data_list = {'default': models.main_model, 'model_list': model_list,
                       'date': datetime.datetime.now().timestamp()}
    save_model_list(model_data_list)
    print('All predictions updated. ' + str(datetime.datetime.now()))
    messaging.notify_predictions_updated()


def multi_model_retrain():
    logging.switch_logging_category(strings.logging_multi_model_retrain)
    for model_name in models.retrain_dense_models:
        retrain(model_name, hyper_parameters.epochs_dense)

    for model_name in models.retrain_lstm_models:
        retrain(model_name, hyper_parameters.epochs_lstm)

    print('All models updated. '+str(datetime.datetime.now()))
    messaging.notify_training_done()
