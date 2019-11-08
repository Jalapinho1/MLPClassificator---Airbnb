from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pca2 import run_pca2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def run_mlp2(normalizedDf):
    X = normalizedDf.drop('price_cat', axis=1)
    y = normalizedDf['price_cat']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scores = []
    mlp = MLPClassifier(hidden_layer_sizes=(19, 45, 3), activation='relu', max_iter=1000,
                        solver='adam', early_stopping=True)
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X):
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        mlp.fit(X_train, y_train)
        plot_learning_curve(mlp, "Crash Learning", X_train, y_train)
        scores.append(mlp.score(X_test, y_test))


    print(np.mean(scores))
    print(cross_val_score(mlp, X, y, cv=10))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(scores)


def run_mlp(normalizedDf,df):
    print(normalizedDf.describe().transpose());

    X = normalizedDf.drop('price_cat', axis=1)
    y = normalizedDf['price_cat']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y)


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    run_pca2(X_train)
    run_pca2(X_test)

    print(X_train.shape)
    print(X_test.shape)

    kf = KFold(n_splits=10)
    mlp = MLPClassifier(hidden_layer_sizes=(19, 30, 5), max_iter=1000, solver='adam', early_stopping=True)

    #Fitting the training data to the network
    mlp.fit(X_train, Y_train)

    # Predicting y for X_test
    predictions = mlp.predict(X_test)
    # predict_train = mlp.predict(X_train)
    cm = confusion_matrix(Y_test, predictions)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))

    print(len(mlp.coefs_[0]))
    print(len(mlp.coefs_[1]))
    print(len(mlp.intercepts_[0]))

    # Printing the accuracy
    # print(cm)
    print("Accuracy of MLPClassifier : ", accuracy(cm))
    print(mlp.score(X_test,Y_test))
    print(cross_val_score(mlp, X, y, cv=10))

