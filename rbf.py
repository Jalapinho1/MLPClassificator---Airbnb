import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Price-normalized]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,0.15])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Price-normalized^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,0.05])
  plt.legend()
  plt.show()

def run_rbf(normalizedDf):
    X = normalizedDf.drop(['price'], axis=1)
    y = normalizedDf[['price']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    print(X_train.shape)
    print(X_test.shape)

    scalerY = MinMaxScaler()
    Y_train = scalerY.fit_transform(Y_train)
    Y_test = scalerY.fit_transform(Y_test)

    model = Sequential()
    model.add(Dense(18, input_dim=18, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='linear'))
    print(model.summary())

    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    # optimizer = Adam(lr=0.0001)
    optimizer = keras.optimizers.RMSprop(0.0001)
    model.compile(loss="mean_squared_error" , optimizer=optimizer,  metrics=['mean_absolute_error', 'mean_squared_error'])

    example_batch = X_train[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    history = model.fit(X_train, Y_train, epochs=1000, validation_split=0.15, callbacks=[earlystopping])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_history(history)

    pred_train = model.predict(X_train)
    print(np.sqrt(mean_squared_error(Y_train, pred_train)))

    pred = model.predict(X_test)
    print(np.sqrt(mean_squared_error(Y_test, pred)))

    print(r2_score(Y_test, pred))

    test_predictions = model.predict(X_test).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(Y_test, test_predictions)
    plt.xlabel('True Values [Price-normalized]')
    plt.ylabel('Predictions [Price-normalized]')
    lims = [0, 0.5]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


