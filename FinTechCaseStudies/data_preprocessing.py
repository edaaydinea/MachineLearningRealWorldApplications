# -*- coding: utf-8 -*-
"""
Data Preprocessing Module

This module contains the DataPreprocessing class which handles the data cleaning,
feature engineering, and transformation steps required before model training.

Created on Fri Feb 7 1:27:20 2025
Created by Eda AYDIN
"""

#### Importing Libraries ####

import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessing:
    """
    A class used to preprocess the dataset for machine learning.

    Attributes
    ----------
    data_path : str
        Path to the dataset file.
    top_screens_path : str
        Path to the top screens file.
    dataset : pd.DataFrame
        The dataset loaded from the data_path.
    top_screens : np.array
        Array of top screens loaded from the top_screens_path.

    Methods
    -------
    load_data():
        Loads the dataset from the specified path.
    initial_cleaning():
        Performs initial cleaning of the dataset.
    plot_histograms():
        Plots histograms of numerical columns in the dataset.
    plot_correlations():
        Plots correlation matrix and bar plot of correlations with the response variable.
    feature_engineering():
        Performs feature engineering on the dataset.
    process_screens():
        Processes the screen_list column and creates new features.
    create_funnels():
        Creates funnel features from the screen data.
    save_dataset(output_path):
        Saves the processed dataset to the specified output path.
    """

    @staticmethod
    def load_data(data_path):
        """Loads the dataset from the specified path."""
        logging.info(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    

    def __init__(self, data_path, top_screens_path):
        """
        Constructs all the necessary attributes for the DataPreprocessing object.

        Parameters
        ----------
        data_path : str
            Path to the dataset file.
        top_screens_path : str
            Path to the top screens file.
        """
        logging.info("Initializing DataPreprocessing object")
        self.data_path = data_path
        self.top_screens_path = top_screens_path
        self.dataset = pd.read_csv(data_path)
        self.top_screens = pd.read_csv(top_screens_path).top_screens.values
        self.graph_path = "../graphs/"

    def initial_cleaning(self):
        """Performs initial cleaning of the dataset."""
        logging.info("Performing initial cleaning of the dataset")
        self.dataset["hour"] = self.dataset.hour.str.slice(1, 3).astype(int)
        return self.dataset

    def plot_histograms(self):
        """Plots histograms of numerical columns in the dataset."""
        logging.info("Plotting histograms of numerical columns")
        dataset2 = self.dataset.copy().drop(columns=['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
        plt.suptitle('Histograms of Numerical Columns', fontsize=20)
        for i in range(1, dataset2.shape[1] + 1):
            plt.subplot(3, 3, i)
            f = plt.gca()
            f.set_title(dataset2.columns.values[i - 1])
            vals = np.size(dataset2.iloc[:, i - 1].unique())
            plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(self.graph_path, 'histograms.jpg'), bbox_inches='tight')
        return self.dataset

    def plot_correlations(self):
        """Plots correlation matrix and bar plot of correlations with the response variable."""
        logging.info("Plotting correlation matrix and bar plot of correlations with the response variable")
        dataset2 = self.dataset.copy().drop(columns=['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
        # Correlation with Response Variable
        dataset2.corrwith(self.dataset.enrolled).plot.bar(figsize=(20, 10),
                                                          title='Correlation with Response variable',
                                                          fontsize=15, rot=45,
                                                          grid=True)
        
        # Correlation Matrix
        sn.set_theme(style="white", font_scale=2)
        
        # Compute the correlation matrix
        corr = dataset2.corr()
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(18, 15))
        f.suptitle("Correlation Matrix", fontsize=40)
        
        # Generate a custom diverging colormap
        cmap = sn.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        plt.savefig(os.path.join(self.graph_path, 'correlation_matrix.jpg'), bbox_inches='tight')
        return self.dataset
        

    def feature_engineering(self):
        """Performs feature engineering on the dataset."""
        logging.info("Performing feature engineering on the dataset")
        self._format_date_columns()
        self._calculate_time_difference()
        self._plot_response_distribution()
        self._plot_response_distribution2()
        self._clean_dataset()
        return self.dataset

    def _format_date_columns(self):
        """Formats the date columns in the dataset."""
        logging.info("Formatting date columns")
        self.dataset["first_open"] = [parser.parse(row_date) for row_date in self.dataset["first_open"]]
        self.dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in self.dataset["enrolled_date"]]

    def _calculate_time_difference(self):
        """Calculates the time difference between first_open and enrolled_date."""
        logging.info("Calculating time difference between first_open and enrolled_date")
        self.dataset["difference"] = (self.dataset.enrolled_date - self.dataset.first_open).astype('timedelta64[ns]') / np.timedelta64(1, 'h')

    def _plot_response_distribution(self):
        """Plots the distribution of time since screen reached."""
        logging.info("Plotting distribution of time since screen reached")
        plt.hist(self.dataset["difference"].dropna(), color='#3F5D7D')
        plt.title('Distribution of Time-Since-Screen-Reached')
        plt.savefig(os.path.join(self.graph_path, 'time_since_screen_reached.jpg'), bbox_inches='tight')
        
    def _plot_response_distribution2(self):
        """Plots the distribution of time since screen reached."""
        logging.info("Plotting distribution of time since screen reached between 0 and 100 hours")
        plt.hist(self.dataset["difference"].dropna(), color='#3F5D7D', range=[0, 100])
        plt.title('Distribution of Time-Since-Screen-Reached-Between-0-and-100-Hours')
        plt.savefig(os.path.join(self.graph_path, 'time_since_screen_reached_0_100.jpg'), bbox_inches='tight')

    def _clean_dataset(self):
        """Cleans the dataset by removing unnecessary columns and handling outliers."""
        logging.info("Cleaning dataset by removing unnecessary columns and handling outliers")
        self.dataset.loc[self.dataset.difference > 48, 'enrolled'] = 0
        self.dataset = self.dataset.drop(columns=['enrolled_date', 'difference', 'first_open'])

    def process_screens(self):
        """Processes the screen_list column and creates new features."""
        logging.info("Processing screen_list column and creating new features")
        
        # Mapping Screens to Fields
        self.dataset["screen_list"] = self.dataset.screen_list.astype(str) + ','
        
        for sc in self.top_screens:
            self.dataset[sc] = self.dataset.screen_list.str.contains(sc).astype(int)
            self.dataset['screen_list'] = self.dataset.screen_list.str.replace(sc + ",", "")
        self.dataset['Other'] = self.dataset.screen_list.str.count(",")
        self.dataset = self.dataset.drop(columns=['screen_list'])
        return self.dataset

    def create_funnels(self):
        """Creates funnel features from the screen data."""
        logging.info("Creating funnel features from the screen data")
        savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
        self.dataset["SavingCount"] = self.dataset[savings_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=savings_screens)

        cm_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
        self.dataset["CMCount"] = self.dataset[cm_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=cm_screens)

        cc_screens = ["CC1", "CC1Category", "CC3"]
        self.dataset["CCCount"] = self.dataset[cc_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=cc_screens)

        loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]
        self.dataset["LoansCount"] = self.dataset[loan_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=loan_screens)
        return self.dataset

    def save_dataset(self, output_path):
        """Saves the processed dataset to the specified output path."""
        logging.info(f"Saving processed dataset to {output_path}")
        self.dataset.to_csv(output_path, index=False)

