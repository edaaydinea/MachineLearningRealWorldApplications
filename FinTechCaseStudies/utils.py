import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def plot_histograms(dataset, title='Histograms of Numerical Columns'):
    plt.suptitle(title, fontsize=20)
    for i in range(1, dataset.shape[1] + 1):
        plt.subplot(3, 3, i)
        f = plt.gca()
        f.set_title(dataset.columns.values[i - 1])
        vals = dataset.iloc[:, i - 1].nunique()
        plt.hist(dataset.iloc[:, i - 1], bins=vals, color='#3F5D7D')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_correlation_matrix(dataset, title='Correlation Matrix'):
    sn.set(style="white", font_scale=2)
    corr = dataset.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(18, 15))
    f.suptitle(title, fontsize=40)
    cmap = sn.diverging_palette(220, 10, as_cmap=True)
    sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5})
