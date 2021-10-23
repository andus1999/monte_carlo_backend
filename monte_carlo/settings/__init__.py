from . import models
from . import hyper_parameters

# model settings
x_train_length = None
forecast_in_days = None
loss = None
investment_condition = None
test_mode: bool = False


def init_settings(model_name):
    global x_train_length
    global forecast_in_days
    global loss
    global investment_condition
    x_train_length, forecast_in_days, loss = models.get_settings(model_name)
    investment_condition = hyper_parameters.investment_condition_factor*forecast_in_days
    print('Settings initialized.')


def clear_settings():
    global x_train_length
    global forecast_in_days
    global loss
    global investment_condition
    x_train_length = None
    forecast_in_days = None
    loss = None
    investment_condition = None
