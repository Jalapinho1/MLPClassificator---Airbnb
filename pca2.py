import pandas as pd
from sklearn import preprocessing
from dataloader import DataLoader
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def run_pca2(normalizedDf):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(normalizedDf)
    # Plot the explained variances
    # features = range(pca.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.show()
    PCA_components = pd.DataFrame(principalComponents)

    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())