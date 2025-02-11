import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from config import HISTOGRAM_DROP_COLS, PIE_CHART_COLS, CORRELATION_DROP_COLS, VISUALIZATION_PATH

def save_plot_to_visualization_path(graph_name = "visualization.png"):
    graph_path = os.path.join(VISUALIZATION_PATH, graph_name)
    plt.savefig(graph_path, bbox_inches='tight')


def plot_histograms(dataset, title='Histograms of Numerical Columns'):
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle(title, fontsize=20)
    for i in range(1, dataset.shape[1] + 1):
        plt.subplot(6, 5, i)
        f = plt.gca()
        f.axes.get_yaxis().set_visible(False)
        f.set_title(dataset.columns.values[i - 1])
        vals = dataset.iloc[:, i - 1].nunique()
        plt.hist(dataset.iloc[:, i - 1], bins=vals, color='#3F5D7D')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot_to_visualization_path(graph_name="histograms.png")

    
def plot_pie_charts(dataset, title='Pie Chart Distributions'):
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle('Pie Chart Distributions', fontsize=20)
    for i in range(1, dataset.shape[1] + 1):
        plt.subplot(6, 3, i)
        f = plt.gca()
        f.axes.get_yaxis().set_visible(False)
        f.set_title(dataset.columns.values[i - 1])
    
        values = dataset.iloc[:, i - 1].value_counts(normalize = True).values
        index = dataset.iloc[:, i - 1].value_counts(normalize = True).index
        plt.pie(values, labels = index, autopct='%1.1f%%')
        plt.axis('equal')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot_to_visualization_path(graph_name="pie_charts.png")

def plot_correlation_matrix(dataset, title='Correlation Matrix with Response Variable'):
    sn.set_theme(style="white")
    numeric_dataset = dataset.drop(columns=HISTOGRAM_DROP_COLS).select_dtypes(include=[np.number])
    corr = numeric_dataset.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(18, 15))
    cmap = sn.diverging_palette(220, 10, as_cmap=True)
    sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5})
    save_plot_to_visualization_path(graph_name="correlation_matrix.png")