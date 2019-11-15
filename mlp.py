import keras
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.optimizers import Adam
from keras.optimizers import SGD

def run_mlp(normalizedDf):
    logging.getLogger('tensorflow').disabled = True

    X = normalizedDf.drop(['cheap', 'medium', 'expensive'], axis=1)
    y = normalizedDf[['cheap', 'medium', 'expensive']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(18, input_dim=18, activation='tanh'))
    model.add(Dense(45, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    print(model.summary())

    model.compile(SGD(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                  patience=2, verbose=0, mode='auto')
    history = model.fit(X_train, Y_train, validation_split=0.15, epochs=500, verbose=2, callbacks=[earlystopping])

    # evaluate the model
    _, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.show()