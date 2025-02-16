import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self, file_path = os.path.join(config.DataConfig.DATA_DIR, 
                                                config.DataConfig.INPUT_DATA_PATH)):
        """Initializes the data preprocessor with the given data path"""
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Loads the data from the given data path"""
        try:
            logging.info(f"Loading data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            
    def preprocess(self):
        """Preprocesses the data by normalizing and saving the processed file"""
        try:
            self.data["normalizedAmount"] = StandardScaler().fit_transform(self.data["Amount"].values.reshape(-1,1))
            self.data = self.data.drop(["Time", "Amount"], axis=1)
            logging.info("Data preprocessed successfully")
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            
    
    def save_processed_data(self, file_path = os.path.join(config.DataConfig.DATA_DIR, 
                                                           config.DataConfig.PROCESSED_DATA_PATH)):
        try:
            logging.info(f"Saving processed data to {file_path}")
            self.data.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")
            raise Exception(f"Error saving processed data to {file_path}: {e}")
                  