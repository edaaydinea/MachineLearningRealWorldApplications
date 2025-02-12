import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataVisualizer:
    def __init__(self, output_dir=config.GRAPH_DIR):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_histograms(self, df, bins=100):
        """
        Plot histograms of numerical columns in the dataset
        """
        num_columns = df.select_dtypes(include=['int64', 'float64']).columns
        n_cols = 3
        n_rows = int(np.ceil(len(num_columns) / n_cols))
        
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        for i, col in enumerate(num_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(df[col].dropna(), bins=bins, color='#3F5D7D')
            plt.title(col)
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'histograms.png')
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Histograms saved to {output_file}")
    
    def plot_correlation_heatmap(self, df):
        """
        Create and save a heatmap of the correlation matrix of the DataFrame.
        """
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=0.3, center=0,
                    square=True, linewidths=0.5)
        
        output_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Correlation heatmap saved to {output_file}")

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Calculate and save the Confusion Matrix as a heatmap.
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        output_file = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Confusion matrix saved to {output_file}")
        return cm

"""
def plot_histograms(dataset):
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle('Histograms of Numerical Columns', fontsize=20)
    for i in range(dataset.shape[1]):
        plt.subplot(6, 3, i + 1)
        f = plt.gca()
        f.set_title(dataset.columns.values[i])
        vals = np.size(dataset.iloc[:, i].unique())
        if vals >= 100:
            vals = 100
        plt.hist(dataset.iloc[:, i], bins=vals, color='#3F5D7D')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correlation_matrix(dataset):
    sn.set(style="white")
    corr = dataset.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(18, 15))
    cmap = sn.diverging_palette(220, 10, as_cmap=True)
    sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=(0, 1), columns=(0, 1))
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()
"""