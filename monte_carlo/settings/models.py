import tensorflow as tf
from . import hyper_parameters
from .. import mc_model
from .. import settings

main_model = 'monte_carlo_lstm3'

dense_models = ['monte_carlo_bnd1', 'monte_carlo_bnd2', 'monte_carlo_bnd3', 'monte_carlo_bnd7']
lstm_models = ['monte_carlo_lstm3', 'monte_carlo_lstm1', 'monte_carlo_lstm2', 'monte_carlo_lstm7']


def get_description(model_name: str) -> str:
    description = ""

    if model_name == 'monte_carlo_bnd1':
        description += 'Dense neural net for one day forecasts.'

    elif model_name == 'monte_carlo_bnd2':
        description += 'Dense neural net for two day forecasts.'

    elif model_name == 'monte_carlo_bnd3':
        description += 'Dense neural net for three day forecasts.'

    elif model_name == 'monte_carlo_bnd7':
        description += 'Dense neural net for seven day forecasts.'

    elif model_name == 'monte_carlo_lstm1':
        description += 'Long short term memory neural net for one day forecasts.'

    elif model_name == 'monte_carlo_lstm2':
        description += 'Long short term memory neural net for two day forecasts.'

    elif model_name == 'monte_carlo_lstm3':
        description += 'Long short term memory neural net for three day forecasts.'

    elif model_name == 'monte_carlo_lstm7':
        description += 'Long short term memory neural net for seven day forecasts.'

    elif model_name == 'monte_carlo_lstmn3':
        description += 'Long short term memory neural net with more parameters for three day forecasts.'

    else:
        description += 'No model description available.'

    description += "\n\nGeneral information:\nThe model performance metric is the model's performance compared to " \
                   "random investments on a per prediction basis (i.e. performance over {forecast:.0f} {check_forecast_plural}). " \
                   "The investment performance is the model's performance over a period of {test_samples:.0f} " \
                   "days compared to random investments over the same period. This metric also takes a spread " \
                   "margin of {spread:.1f} % per trade into account. To achieve similar results, only invest " \
                   "in predictions greater than {condition:.0f} %. The correlation metric shows " \
                   "pearson correlation on the current test set and the highest historical correlation achieved on " \
                   "any validation set in that order.".format(forecast=settings.forecast_in_days,
                                                              check_forecast_plural=check_plural('day',
                                                                                                 settings.forecast_in_days),
                                                              test_samples=hyper_parameters.test_sample_size,
                                                              spread=hyper_parameters.spread_margin * 100,
                                                              condition=settings.investment_condition * 100)
    return description


def build_model(x_train, m_train, y_train, model_name):
    ''' depreciated
    if model_name == 'cm_v9':
        "Batch norm and dropout."
        neurons = 128
        drop = 0.5
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        y_shape = y_train.shape[-1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop // 2)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dense(neurons)(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(y_shape)(x)

        output = tf.keras.layers.Reshape((1, y_shape))(x)

        dnn_model = tf.keras.Model([input_hist, input_meta], output)
    # depreciated'''

    if model_name == 'monte_carlo':
        """Dropout model."""
        neurons = 128
        drop = 0.5

        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]

        def activation(val):
            return tf.keras.activations.relu(val, alpha=0.1)

        input_hist = tf.keras.layers.Input(shape=x_shape)
        x = tf.keras.layers.Dense(neurons, activation=activation)(input_hist)
        x = tf.keras.layers.Dropout(drop // 2)(x, training=True)
        x = tf.keras.layers.Dense(neurons, activation=activation)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        output_hist = tf.keras.layers.Dense(neurons, activation=activation)(x)

        input_meta = tf.keras.layers.Input(shape=m_shape)
        x = tf.keras.layers.Dense(neurons // 2, activation=activation)(input_meta)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        output_meta = tf.keras.layers.Dense(neurons // 2, activation=activation)(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dropout(drop)(combined, training=True)
        x = tf.keras.layers.Dense(neurons * 4, activation=activation)(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 4, activation=activation)(x)
        output = tf.keras.layers.Dense(1, activation=activation)(x)

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    elif model_name == 'monte_carlo_bn':
        """Batch norm before activation. Standard output neurons."""
        neurons = 128
        drop = 0.5
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dense(neurons * 2)(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        output = tf.keras.layers.Dense(1)(x)

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    elif model_name == 'monte_carlo_bnd1' or model_name == 'monte_carlo_bnd2' or model_name == 'monte_carlo_bnd3' or model_name == 'monte_carlo_bnd7':
        """Batch norm before activation. With more dropout layers."""
        neurons = 128
        drop = 0.2
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dropout(drop)(combined, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        output = tf.keras.layers.Dense(1)(x)

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    elif model_name == 'monte_carlo_bndl':
        """Batch norm before activation. With more dropout layers."""
        neurons = 128
        drop = 0.2
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dropout(drop)(combined, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(1, activation='softsign')(x)
        output = tf.keras.layers.add([x, tf.ones_like(x)])

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    elif model_name == 'monte_carlo_bndc':
        """Batch norm before activation. With more dropout layers."""
        neurons = 128
        drop = 0.2
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dropout(drop)(combined, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(1, activation='softsign')(x)
        output = tf.keras.layers.add([x, tf.ones_like(x)])

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    elif model_name == 'monte_carlo_lstm1' or model_name == 'monte_carlo_lstm2' or model_name == 'monte_carlo_lstm3' or model_name == 'monte_carlo_lstm7':
        """LSTM model. Batch norm before activation. With more dropout layers."""
        neurons = 128
        drop = 0.2
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.LSTM(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dropout(drop)(combined, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        output = tf.keras.layers.Dense(1)(x)

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    elif model_name == 'monte_carlo_lstmn3':
        """LSTM model. Batch norm before activation. With more dropout layers. More lstm neurons."""
        neurons = 128
        drop = 0.2
        x_shape = x_train.shape[1:3]
        m_shape = m_train.shape[1]
        input_hist = tf.keras.layers.Input(shape=x_shape, name='input_hist_data')
        x = tf.keras.layers.Dense(neurons)(input_hist)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.LSTM(neurons * 4)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_hist = tf.keras.layers.LeakyReLU()(x)

        input_meta = tf.keras.layers.Input(shape=m_shape, name='input_m_data')
        x = tf.keras.layers.Dense(neurons // 2)(input_meta)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons // 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_meta = tf.keras.layers.LeakyReLU()(x)

        combined = tf.keras.layers.concatenate([output_hist, output_meta])
        x = tf.keras.layers.Dropout(drop)(combined, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop)(x, training=True)
        x = tf.keras.layers.Dense(neurons * 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        output = tf.keras.layers.Dense(1)(x)

        dnn_model = tf.keras.Model([input_hist, input_meta], output)

    else:
        raise ValueError('No Model found: ' + model_name)

    dnn_model.summary()
    print('Model created. ' + model_name)
    return dnn_model


def get_settings(model_name):
    loss = tf.keras.losses.Huber(delta=hyper_parameters.huber_loss_delta)
    if model_name == 'monte_carlo_bndl':
        x_train_length = 200
        forecast_in_days = 3
        loss = mc_model.gain
    elif model_name == 'monte_carlo_bndc':
        x_train_length = 200
        forecast_in_days = 3
        loss = mc_model.correlation
    elif model_name == 'cm_v9':
        x_train_length = 365
        forecast_in_days = 7
    elif model_name == 'monte_carlo_bnd7':
        x_train_length = 200
        forecast_in_days = 7
    elif model_name == 'monte_carlo_bnd1':
        x_train_length = 200
        forecast_in_days = 1
    elif model_name == 'monte_carlo_bnd2':
        x_train_length = 200
        forecast_in_days = 2
    elif model_name == 'monte_carlo_bnd3':
        x_train_length = 200
        forecast_in_days = 3
    elif model_name == 'monte_carlo_lstm1':
        x_train_length = 200
        forecast_in_days = 1
    elif model_name == 'monte_carlo_lstm2':
        x_train_length = 200
        forecast_in_days = 2
    elif model_name == 'monte_carlo_lstm3':
        x_train_length = 200
        forecast_in_days = 3
    elif model_name == 'monte_carlo_lstm7':
        x_train_length = 200
        forecast_in_days = 7
    elif model_name == 'monte_carlo_lstmn3':
        x_train_length = 200
        forecast_in_days = 3
    else:
        raise ValueError('No settings found for model: ' + model_name)
    return x_train_length, forecast_in_days, loss


def check_plural(string: str, count: int) -> str:
    if string == 'day':
        if count > 1:
            return 'days'
        else:
            return 'day'
