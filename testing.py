from monte_carlo import mc_main, settings
from data import coinmarketcap
from monte_carlo.utils import messaging, logging
from monte_carlo.settings import models, hyper_parameters
import package_manager


def initialize_test_settings():
    models.lstm_models = [models.main_model]
    models.dense_models = [models.dense_models[0]]
    hyper_parameters.epochs_lstm = 1
    hyper_parameters.epochs_dense = 1
    settings.test_mode = True


def test_routine():
    try:
        package_manager.reload_packages()
        initialize_test_settings()
        coinmarketcap.update()
        mc_main.multi_model_predict()
        logging.stop_logging()
    except Exception as e:
        messaging.notify_exception()
        logging.stop_logging()
        raise e
