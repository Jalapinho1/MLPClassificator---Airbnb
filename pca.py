import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def run_pca(normalizedDf, originalDf):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(normalizedDf.values)
    principalDf = pd.DataFrame(data=principalComponents)
    finalDf = pd.concat([principalDf, originalDf[['price']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA - Data', fontsize=20)
    targets = originalDf.price.unique()
    print(targets)
    colors = ['r', 'g', 'b', 'y', 'c', 'deepskyblue', 'm', 'yellow', 'k', 'darkorange']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['price'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 0]
                   , finalDf.loc[indicesToKeep, 1]
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())