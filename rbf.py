from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pca2 import run_pca2

def run_rbf(normalizedDf,df):
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

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    # Train the model using the training sets
    clf.fit(X_train, Y_train)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(Y_test, y_pred, pos_label='positive', average='micro'))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(Y_test, y_pred, pos_label='positive', average='micro'))



