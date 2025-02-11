# -*- coding: utf-8 -*-
"""
Data Preprocessing Module

This module contains the DataPreprocessing class which handles the data cleaning,
feature engineering, and transformation steps required before model training.

Created on Fri Feb 7 1:27:20 2025
Created by Eda AYDIN
"""

#### Importing Libraries ####

import os
import pandas as pd
import numpy as np
import logging
from utils import plot_histograms, plot_pie_charts, plot_correlation_matrix
from config import HISTOGRAM_DROP_COLS, PIE_CHART_COLS, CORRELATION_DROP_COLS, DATA_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessing:
    """
    A class used to preprocess the dataset for machine learning.

    Attributes
    ----------
    data_path : str
        Path to the dataset file.
    dataset : pd.DataFrame
        The dataset loaded from the data_path.

    Methods
    -------
    load_data():
        Loads the dataset from the specified path.
    initial_cleaning():
        Performs initial cleaning of the dataset.
    feature_engineering():
        Performs feature engineering on the dataset.
    save_dataset(output_path):
        Saves the processed dataset to the specified output path.
    """
    
    @staticmethod 
    def load_data(data_path):
        """Loads the dataset from the specified path."""
        input_data_path = os.path.join(DATA_FOLDER, data_path)
        logging.info(f"Loading dataset from {input_data_path}")
        return pd.read_csv(input_data_path)

    def __init__(self, data_path):
        """
        Constructs all the necessary attributes for the DataPreprocessing object.

        Parameters
        ----------
        data_path : str
            Path to the dataset file.
        """
        logging.info("Initializing DataPreprocessing object")
        input_data_path = os.path.join(DATA_FOLDER, data_path)
        self.dataset = pd.read_csv(input_data_path)
       

    def initial_cleaning(self):
        """Performs initial cleaning of the dataset."""
        logging.info("Performing initial cleaning of the dataset")
        self.dataset = self.dataset[self.dataset['credit_score'] >= 300]
        self.dataset = self.dataset.drop(columns=['credit_score', 'rewards_earned'])
        return self.dataset
    
    def visualize_data(self):
        """Visualizes the dataset using histograms, pie charts, and correlation matrix."""
        logging.info("Visualizing the dataset")
        
        self.dataset2 = self.dataset.drop(columns=HISTOGRAM_DROP_COLS)
        plot_histograms(self.dataset2)
        logging.info("Histograms plotted")
        
        self.dataset2 = self.dataset[PIE_CHART_COLS]
        plot_pie_charts(self.dataset2)
        logging.info("Pie charts plotted")
        
        plot_correlation_matrix(self.dataset)
        logging.info("Correlation matrix plotted")
        self.dataset = self.dataset.drop(columns=['app_web_user'])

    def save_dataset(self, output_path):
        """Saves the processed dataset to the specified output path."""
        logging.info(f"Saving processed dataset to {output_path}")
        self.dataset.to_csv(output_path, index=False)
