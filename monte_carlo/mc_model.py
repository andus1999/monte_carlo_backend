import datetime
import os
import pickle

import numpy as np
import tensorflow as tf

from . import settings
from .settings import hyper_parameters, models
from .utils import callbacks
from .resources import filepaths


def load_model(model_name):
    if model_name == 'monte_carlo_bndl':
        dnn_model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../../cm_model/' + model_name),
                                               custom_objects={'correlation': correlation, 'gain': gain})
    else:
        dnn_model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../../cm_model/' + model_name),
                                               custom_objects={'correlation': correlation})
    print('Model loaded. ' + model_name)
    return dnn_model


def save_model(dnn_model, model_name):
    if settings.test_mode is True:
        print('Models are not saved in test mode.')
        return
    dnn_model.save(os.path.join(os.path.dirname(__file__), '../../cm_model/' + model_name))
    print('Model saved. ' + model_name)


def correlation(y_true, y_pred):
    y_true = tf.reshape(y_true, [tf.size(y_true)])
    y_pred = tf.reshape(y_pred, [tf.size(y_pred)])
    y_true_mean = tf.math.multiply(tf.ones_like(y_true), tf.math.reduce_mean(y_true))
    y_pred_mean = tf.math.multiply(tf.ones_like(y_pred), tf.math.reduce_mean(y_pred))
    y_true_deviations = tf.math.subtract(y_true, y_true_mean)
    y_pred_deviations = tf.math.subtract(y_pred, y_pred_mean)
    pearson_correlation_coefficient = tf.math.divide_no_nan(
        tf.math.reduce_sum(tf.math.multiply(y_true_deviations, y_pred_deviations)),
        tf.math.sqrt(tf.math.multiply(tf.math.reduce_sum(tf.math.square(y_true_deviations)),
                                      tf.math.reduce_sum(tf.math.square(y_pred_deviations)))))
    return tf.negative(pearson_correlation_coefficient)


def gain(y_true, y_pred):
    y_true = tf.reshape(y_true, [tf.size(y_true)])
    investments = tf.subtract(tf.reshape(y_pred, [tf.size(y_pred)]), tf.constant(1.))
    signs = tf.cast(tf.math.sign(investments), tf.float32)
    pos = tf.divide(tf.math.add(signs, tf.constant(1.)), tf.constant(2.))
    neg = tf.divide(tf.math.subtract(signs, tf.constant(1.)), tf.constant(2.))
    investment = tf.math.reduce_sum(
        tf.math.add(tf.multiply(investments, pos), tf.multiply(tf.multiply(investments, y_true), neg)))
    value = tf.reduce_sum(tf.add(tf.multiply(tf.multiply(investments, y_true), pos), tf.multiply(investments, neg)))
    return tf.add(tf.multiply(tf.math.divide_no_nan(value, investment), tf.constant(-1.)), tf.constant(1.))


def save_correlation(model_name, best_correlation):
    path = os.path.join(os.path.dirname(__file__), filepaths.correlation_data_path+model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path+filepaths.correlation_data, 'wb') as f:
        pickle.dump(best_correlation, f, pickle.DEFAULT_PROTOCOL)


def get_correlation(model_name):
    path = os.path.join(os.path.dirname(__file__), filepaths.correlation_data_path+model_name+filepaths.correlation_data)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            file = pickle.load(f)
        return file
    else:
        return None


def fit_model(dnn_model, x_train, m_train, y_train, x_val, m_val, y_val, model_name, epochs, learning_rate,
              tensor_board, patience, baseline=None, checkpoint=False):
    print('Epochs: '+str(epochs)+' Patience: '+str(patience))
    if baseline is not None:
        print('val_correlation baseline: ' + "{:.3f}".format(baseline))
    else:
        print('val_correlation baseline: ' + str(baseline))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if dnn_model.optimizer is not None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        weights = dnn_model.optimizer.weights
        print('Applying optimizer weights.')
        optimizer.set_weights(weights)

    dnn_model.compile(optimizer=optimizer, loss=settings.loss, metrics=[correlation])

    batch_size = 128
    callbacks_list = []

    # Model checkpoint callback
    if checkpoint:
        batches_per_epoch = len(x_train) // batch_size + 1
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.path.dirname(__file__), '../../cm_model/' + model_name),
            verbose=1,
            save_freq=batches_per_epoch * 20)
        callbacks_list.append(cp_callback)

    # Early stopping callback
    if model_name == 'monte_carlo_bndl' or model_name == 'monte_carlo_bndc':
        es_callback = callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                              restore_best_weights=True, mode="min",
                                              verbose=1)
    else:
        es_callback = callbacks.EarlyStopping(monitor='val_correlation', patience=patience,
                                              restore_best_weights=True, mode="min",
                                              verbose=1, baseline=baseline)
    callbacks_list.append(es_callback)

    # Tensorboard callback
    if tensor_board:
        print('Using Tensorboard callback. --logdir logs/fit')
        log_dir = "logs/fit/" + model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks_list.append(tb_callback)

    validation_data = None
    if x_val is not None and m_val is not None and y_val is not None:
        validation_data = ([x_val, m_val], y_val)

    dnn_model.fit(x=[x_train, m_train], y=y_train,
                  validation_data=validation_data, epochs=epochs,
                  shuffle=True, callbacks=callbacks_list, batch_size=batch_size, verbose=2)
    best_correlation = es_callback.best
    print('Best correlation: '+str(round(best_correlation, 3)))
    save_correlation(model_name, best_correlation)
    return dnn_model


def train_model(x_train, m_train, y_train, x_val, m_val, y_val, model_name, epochs=10, learning_rate=0.0001,
                tensor_board=False):
    dnn_model = models.build_model(x_train, m_train, y_train, model_name)
    print('Training model.')
    dnn_model = fit_model(dnn_model, x_train, m_train, y_train, x_val, m_val, y_val, model_name, epochs=epochs,
                          learning_rate=learning_rate, tensor_board=tensor_board,
                          patience=hyper_parameters.train_patience)
    save_model(dnn_model, model_name)
    return dnn_model


def retrain_model(x_train, m_train, y_train, x_val, m_val, y_val, model_name, epochs=10, learning_rate=0.0001,
                  tensor_board=False):
    dnn_model = load_model(model_name)
    val_cor = get_correlation(model_name)
    print('Retraining model.')
    patience = np.ceil(epochs*hyper_parameters.retrain_patience_factor)

    dnn_model = fit_model(dnn_model, x_train, m_train, y_train, x_val, m_val, y_val, model_name,
                          epochs=epochs,
                          learning_rate=learning_rate, tensor_board=tensor_board, patience=patience, baseline=val_cor)
    save_model(dnn_model, model_name)
    return dnn_model
