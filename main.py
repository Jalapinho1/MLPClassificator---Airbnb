import pandas as pd
from scipy.spatial.distance import pdist, squareform
from dataloader import DataLoader
from normalizer import normalize_csv
from pca import run_pca
from mlp import run_mlp
from mlp import run_mlp2
from rbf import run_rbf
import matplotlib.pyplot as plt

def normalize_data():
    dataLoader = DataLoader("barcelona2.csv", ";", 0, 500, 0, 16, False)
    df = dataLoader.dataframe
    print("\nRaw Data")
    dataLoader.printDataInfo()
    # dataLoader.printWideData()

    normalizedDf = normalize_csv(df)
    print("\n\nNormalized Data")
    normalizedDf.info()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(normalizedDf)
    # dataLoader.printWideData()


    pd.set_option('display.width', 600)
    pd.set_option('display.max_columns', 20)

def main():
    df = pd.read_csv('BarcelonaLong.csv', nrows=1600, sep=";", engine='python')
    print(df.info())

    normalizedDf = normalize_csv(df)
    print(normalizedDf.info())

    plt.clf()
    normalizedDf.groupby('price_cat').size().plot(kind='bar')
    # plt.show()

    run_mlp2(normalizedDf)
    # run_mlp(normalizedDf, df)
    # run_rbf(normalizedDf, df)
main()