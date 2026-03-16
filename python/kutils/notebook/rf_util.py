from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


def rolling_random_forest(x, y, train_window=5 * 252, retrain_step=252, **kwargs):
    """
    Rolling random forest classifier.
    """


    n_samples = x.shape[0]
    n_train = train_window

    y_pred_prob = []
    y_pred = []
    for i in range(0, n_samples - retrain_step, retrain_step):
        print(i)
        X_train = x.iloc[i:i + n_train]
        y_train = y.iloc[i:i + n_train]
        X_test = x.iloc[i + n_train: i + n_train + retrain_step]
        if X_test.empty or X_train.empty:
            continue
        print(f'Training on {X_train.index[0].strftime("%Y-%m-%d")} '
              f'- {X_train.index[-1].strftime("%Y-%m-%d")} - N_train = {train_window}')
        print(f'Testing on {X_test.index[0].strftime("%Y-%m-%d")} '
              f'- {X_test.index[-1].strftime("%Y-%m-%d")} - N_pred = {retrain_step}')
        model = RandomForestClassifier(**kwargs).fit(X=X_train.values, y=y_train.values)
        prob_ts = pd.Series(data=model.predict_proba(X_test)[:, 1], index=X_test.index, name='pred_prob')
        pred_ts = pd.Series(data=model.predict(X_test), index=X_test.index, name='pred')

        y_pred_prob.append(prob_ts)
        y_pred.append(pred_ts)

        # return pd.Series(np.concatenate(y_pred), index=x.index[n_train::n_test])
    return pd.concat(y_pred_prob, axis=0), pd.concat(y_pred, axis=0)


def ma_crossover(price_ts, short_window, long_window, type='simple', cross_bool=True):
    if type == 'simple':
        ts = (price_ts.rolling(short_window).mean() - price_ts.rolling(long_window).mean())
        if cross_bool:
            return ts.apply(lambda x: int(x > 0))
        else:
            return ts
    elif type == 'ewma':
        ts = (price_ts.ewm(halflife=short_window).mean() - price_ts.ewm(halflife=long_window).mean())
        if cross_bool:
            return ts.apply(lambda x: int(x > 0))
        else:
            return ts
    else:
        raise ValueError(f'ma_crossover type {type} not implemented')


def calculate_rsi(prices, window=14):
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def rsi_to_signals(rsi, upper=70, lower=30):
    """
    Convert RSI values to buy/sell signals.
    Buy signal when RSI crosses above upper threshold, sell signal when it crosses below lower threshold.
    """
    long_signal = (rsi > upper).astype(int).diff().fillna(0)
    short_signal = (rsi < lower).astype(int).diff().fillna(0)

    # Only care about when it turns on
    long_signal = long_signal.mask(long_signal < 1).fillna(0)
    short_signal = short_signal.mask(short_signal < 1).fillna(0)

    return long_signal, short_signal


def print_eval_metrics(y_truth, y_pred, y_positive):

    accuracy = accuracy_score(y_true=y_truth.values, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.2%}')

    # precision is more important
    precision = precision_score(y_true=y_truth.values, y_pred=y_pred)
    print(f'Precision: {precision:.2%}')

    # "practical" as in the forward return being > 0
    practical_precision = precision_score(y_true=y_positive.loc[y_pred.index].values, y_pred=y_pred)
    print(f'Practical Precision: {practical_precision:.2%}')