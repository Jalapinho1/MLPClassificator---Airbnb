import pandas as pd
from scipy.spatial.distance import pdist, squareform
from normalizer import normalize_csv
from normalizer import normalize_csv_regression
import matplotlib.pyplot as plt
from mlp import run_mlp
from rbf import run_rbf
from pca2 import run_pca2

def run_classificator(df):
    normalizedDf = normalize_csv(df)
    print(normalizedDf.info())
    run_pca2(normalizedDf)
    run_mlp(normalizedDf)

def run_regressor(df):
    normalizedDf = normalize_csv_regression(df)
    print(normalizedDf.info())
    run_pca2(normalizedDf)
    run_rbf(normalizedDf)


def main():
    df = pd.read_csv('BarcelonaLong.csv', nrows=5000, sep=";", engine='python')
    print(df.info())

    # plt.clf()
    # df.groupby('price_cat').size().plot(kind='bar')
    # plt.show()

    run_classificator(df)
    # run_regressor(df)

main()