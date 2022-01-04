import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.colors import SymLogNorm
from . import mc_prediction
from . import mc_model
from . import settings
from .settings import hyper_parameters


def get_percentage_gain(y_true, y_pred):
    true_gains = []
    predicted_gains = []
    loss = settings.loss(y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32))
    for i in range(len(y_pred)):
        pre = y_pred[i]
        tr = y_true[i]
        predicted_gain = pre - 1
        true_gain = tr - 1
        true_gains.append(true_gain)
        predicted_gains.append(predicted_gain)
    return np.array(true_gains), np.array(predicted_gains), loss


def get_bayesian_gain(dnn_model, x_data, m_data, y_true, bayesian_sets):
    data_y_list = np.empty([bayesian_sets, len(y_true)])
    predictions = np.empty([bayesian_sets, len(y_true)])

    for i in range(bayesian_sets):
        data_y_list[i] = y_true
        predictions[i] = dnn_model.predict([x_data, m_data]).flatten()

    return get_percentage_gain(np.concatenate(data_y_list),
                               np.concatenate(predictions))


def scatter_plot(true, pred, title, bayesian_sets, log_path, dot_size=1.):
    cor = np.corrcoef(true, pred)

    lims = [-0.2, 0.3]
    plt.figure(figsize=(7, 7))
    plt.axes(aspect='equal')
    size = len(true) // bayesian_sets
    gradient = np.empty([len(true)])
    for i in range(bayesian_sets):
        gradient[i * size:(i + 1) * size] = range(0, size)
    gradient.flatten()

    color_map = cm.get_cmap('viridis')
    plt.hist2d(true, pred, 100, range=[lims, lims], cmap=color_map, norm=SymLogNorm(1))
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title + ' ' + str(round(cor[0][1], 5)))

    _ = plt.plot(lims, lims, color='grey')
    plt.xlim(lims)
    plt.ylim(lims)
    if log_path is None:
        plt.show()
    else:
        plt.savefig(log_path + title + '.png')
    plt.close()

    print('Correlation ' + title + ': ' + str(round(cor[0][1], 3)))
    # print('Investing algorithm: ' + str(investing_algorithm(true, pred, bayesian_sets)))
    return cor[0][1]


def invest(pre):
    return pre


def investing_algorithm(y_true, y_pred, title, bayesian_sets, log_path):
    investment_condition = settings.forecast_in_days*hyper_parameters.investment_condition_factor
    spread_loss = 1 - hyper_parameters.spread_margin
    num_coins = len(y_pred) // hyper_parameters.test_sample_size // bayesian_sets
    top_k = num_coins//10*bayesian_sets
    set_size = num_coins * hyper_parameters.test_sample_size
    print('Investing algorithm.')
    print('Number of coins: ' + str(num_coins))
    print('Investing in predictions greater than {:.2f}'.format(investment_condition))

    # get average predictions
    daily_true_values = np.empty([num_coins, hyper_parameters.test_sample_size, bayesian_sets])
    daily_predictions = np.empty([num_coins, hyper_parameters.test_sample_size, bayesian_sets])
    for coin in range(num_coins):
        for day in range(hyper_parameters.test_sample_size):
            true_values = np.empty([bayesian_sets])
            predicted_values = np.empty([bayesian_sets])
            for bayesian_set in range(bayesian_sets):
                index = day + coin * hyper_parameters.test_sample_size + bayesian_set * set_size
                true_values[bayesian_set] = y_true[index]
                predicted_values[bayesian_set] = y_pred[index]
            daily_true_values[coin, day, :] = true_values
            assert(np.std(true_values) == 0)
            daily_predictions[coin, day, :] = predicted_values

    # algorithm
    investment_timelines = settings.forecast_in_days
    scale_factor = num_coins//50
    capital = hyper_parameters.capital * scale_factor / settings.forecast_in_days
    investment = hyper_parameters.investment / settings.forecast_in_days
    print('Capital: {} Investment: {}'.format(capital, investment))
    values = np.ones([investment_timelines, hyper_parameters.test_sample_size+1])
    random_values = np.ones([investment_timelines, hyper_parameters.test_sample_size+1])
    for day in range(hyper_parameters.test_sample_size):
        investment_timeline = day % investment_timelines
        true_values_selected = []
        predicted_values_selected = []
        for coin in range(num_coins):
            prediction = np.average(daily_predictions[coin, day])
            true_value = np.average(daily_true_values[coin, day])
            if prediction > investment_condition:
                true_values_selected.append(true_value)
                predicted_values_selected.append(prediction)
        # top_k_indices = math_utils.top_k_indices(daily_predictions[:, day, :].flatten(), top_k)
        # true_values_selected = daily_true_values[:, day, :].flatten()[top_k_indices]
        if len(true_values_selected) == 0:
            gain = 1
        else:
            investments = []
            values_after_investment = []
            for i in range(len(predicted_values_selected)):
                prediction = predicted_values_selected[i]
                true_value = true_values_selected[i]
                investment_for_prediction = investment/investment_condition*prediction
                investments.append(investment_for_prediction)
                values_after_investment.append(investment_for_prediction*(true_value+1)*spread_loss)
            if sum(investments) > capital:
                factor = capital/sum(investments)
                investments = [n * factor for n in investments]
                values_after_investment = [n * factor for n in values_after_investment]
                assert math.isclose(capital, sum(investments))
                gain = sum(values_after_investment)/sum(investments)
            else:
                portfolio_after_investment = sum(values_after_investment)+capital-sum(investments)
                gain = portfolio_after_investment / capital

        random_gain = np.average(daily_true_values[:, day, :])+1
        current_value = values[investment_timeline, day]
        current_random_value = random_values[investment_timeline, day]
        new_value = current_value*gain
        new_random_value = current_random_value*random_gain
        values[investment_timeline, day+1:] = new_value
        random_values[investment_timeline, day+1:] = new_random_value

    average_values = np.average(values, axis=0)
    average_random_values = np.average(random_values, axis=0)

    plt.figure(figsize=(7, 5))
    plt.title(title)
    legend_entries = ['Values']
    plt.plot(range(len(average_values)), average_values)
    legend_entries.append('Random Investment')
    plt.plot(range(len(average_random_values)), average_random_values)
    plt.legend(legend_entries, loc='upper left', shadow=True)
    if log_path is None:
        plt.show()
    else:
        plt.savefig(log_path + title + '.png')
    plt.close()
    print('Gain over set: ' + str(average_values[-1]))
    print('Random gain over set: ' + str(average_random_values[-1]))
    performance = average_values[-1] / average_random_values[-1]
    if math.isnan(performance):
        performance = 0
    print('Investing performance: ' + str(round(performance, 3)))
    return performance


def benchmark(true, pred, title, bayesian_sets, log_path):
    print('\n')
    print(title)
    inv = 0
    val = 0
    binv = 0
    bval = 0
    len_benchmark = 20
    increment = 0.02
    invs = np.zeros(len_benchmark)
    vals = np.zeros(len_benchmark)
    gains = np.zeros(len_benchmark)
    counts = np.zeros(len_benchmark)

    for i in range(len(pred)):
        pre = pred[i]
        tr = true[i]
        if pre > 0:
            inv += invest(pre)
            val += (tr + 1) * invest(pre)
        for j in range(len_benchmark):
            if pre > j * increment:
                invs[j] += invest(pre)
                vals[j] += (tr + 1) * invest(pre)
                counts[j] += 1
        binv += 1
        bval += tr + 1

    for j in range(len_benchmark):
        if invs[j] == 0:
            gains[j] = 0
        else:
            gains[j] = (vals[j] / invs[j] - 1) * 100

    b_gain = (mc_prediction.save_div(bval, binv) - 1) * 100
    gain = (mc_prediction.save_div(val, inv) - 1) * 100

    model_performance = (gain + 100) / (b_gain + 100)

    x = np.arange(0, len_benchmark * increment, increment)
    plt.figure(figsize=(10, 5))
    plt.plot(x, gains)
    plt.plot(x, np.repeat(b_gain, len(x)))
    for i, txt in enumerate(counts):
        plt.annotate(round(txt), (x[i], gains[i]))
    plt.title(title)
    if log_path is None:
        plt.show()
    else:
        plt.savefig(log_path + title + '.png')
    plt.close()
    print('Test: ' + str(round(gain, 1)))
    print('Random: ' + str(round(b_gain, 1)))

    investing_performance = investing_algorithm(true, pred, title + ' Investment Algorithm', bayesian_sets, log_path)

    return model_performance, investing_performance


def get_gain(tab, days):
    value_today = tab[-1 - days, 3]
    value_future = tab[-1, 3]
    gain = value_future / value_today - 1
    return gain


def evaluate_model(dnn_model, model_name, tickers_for_data, x_train, m_train, y_train, x_val, m_val, y_val, x_test,
                   m_test, y_test,
                   log_path=None):
    """Returns a list of model performance, correlation, train loss, validation loss and test loss in that order.
    Model performance and correlation are calculated using the coinbase coins test set."""

    bayesian_sets = hyper_parameters.bayesian_sets
    print('Test samples: ' + str(hyper_parameters.test_sample_size))

    val_true, val_pred, val_loss = None, None, 0.
    val_cor = -mc_model.get_correlation(model_name)
    if x_val is not None and m_val is not None and y_val is not None:
        val_true, val_pred, val_loss = get_bayesian_gain(dnn_model, x_val, m_val, y_val, bayesian_sets)
        scatter_plot(val_true, val_pred, title='Validation Set', bayesian_sets=bayesian_sets,
                     log_path=log_path)

    test_true, test_pred, test_loss = get_bayesian_gain(dnn_model, x_test, m_test, y_test, bayesian_sets)
    scatter_plot(test_true, test_pred, title='Test Set', bayesian_sets=bayesian_sets, log_path=log_path)
    x_test_cb = []
    m_test_cb = []
    y_test_cb = []
    for i in range(0, len(tickers_for_data)):
        cb = tickers_for_data[i][3]
        for j in range(0, hyper_parameters.test_sample_size):
            if cb:
                index = i * hyper_parameters.test_sample_size + j
                x_test_cb.append(x_test[index])
                m_test_cb.append(m_test[index])
                y_test_cb.append(y_test[index])
    x_test_cb = np.array(x_test_cb)
    m_test_cb = np.array(m_test_cb)
    y_test_cb = np.array(y_test_cb)

    test_true_cb, test_pred_cb, test_loss_cb = get_bayesian_gain(dnn_model, x_test_cb, m_test_cb, y_test_cb,
                                                                 bayesian_sets)
    correlation = scatter_plot(test_true_cb, test_pred_cb, title='Test Set Coinbase', bayesian_sets=bayesian_sets,
                               log_path=log_path)

    train_loss = 0.
    if x_train is not None and m_train is not None and y_train is not None:
        y_train_pred = dnn_model.predict([x_train, m_train], verbose=1).flatten()
        train_true, train_pred, train_loss = get_percentage_gain(y_train, y_train_pred)
        scatter_plot(train_true, train_pred, title='Train Set', bayesian_sets=1, dot_size=0.1, log_path=log_path)

    if val_true is not None and val_pred is not None:
        benchmark(val_true, val_pred, 'Validation Set Benchmark', bayesian_sets, log_path=log_path)
    benchmark(test_true, test_pred, 'Test Set Benchmark', bayesian_sets, log_path=log_path)
    model_performance, investing_performance = benchmark(test_true_cb, test_pred_cb, 'Test Set Benchmark Coinbase',
                                                         bayesian_sets, log_path=log_path)

    print('\nSummary:')
    print('Spread margin used: ' + str(hyper_parameters.spread_margin))
    print('Model performance: ' + str(round(model_performance, 3)))
    print('Investing performance: ' + str(round(investing_performance, 3)))
    print('Correlation: ' + str(round(correlation, 3)))
    if train_loss != 0.:
        train_loss = train_loss.numpy().item()
        print('Train loss: ' + str(round(train_loss, 5)))
    if val_loss != 0.:
        val_loss = val_loss.numpy().item()
        print('Validation loss: ' + str(round(val_loss, 5)))
    print('Test loss: ' + str(round(test_loss.numpy().item(), 5)))

    return model_performance, investing_performance, correlation, val_cor, train_loss, val_loss, test_loss.numpy().item()
