import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import config
import numpy as np
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, model_type = None):
        """Initializes the model trainer class"""
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = pd.read_csv(os.path.join(config.DataConfig.DATA_DIR, 
                                            config.DataConfig.PROCESSED_DATA_PATH))
        
    def data_splitting(self):
        """Loads preprocessed data and splits it into training and testing sets"""
        try:
            X = self.data.iloc[:, self.data.columns != 'Class']
            y = self.data.iloc[:, self.data.columns == 'Class']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=config.ModelConfig.TEST_SIZE, random_state=config.ModelConfig.RANDOM_STATE)
            logging.info("Data loaded and split into training and testing sets")
        except Exception as e:
            logging.error(f"Error in splitting data: {str(e)}")
    
    def training(self):
        """Trains the model with the given training data"""
        try:
            if self.model_type == "decision_tree":
                self.model = DecisionTreeClassifier(random_state=config.ModelConfig.RANDOM_STATE)
            elif self.model_type == "random_forest":
                self.model = RandomForestClassifier(random_state=config.ModelConfig.RANDOM_STATE)
            elif self.model_type == "xgboost":
                self.model = XGBClassifier(use_label_encoder=False, 
                                           eval_metric=config.XgboostConfig.EVAL_METRIC, 
                                           random_state=config.ModelConfig.RANDOM_STATE)
            elif self.model_type == "lightgbm":
                self.model = lgb.LGBMClassifier(random_state=config.ModelConfig.RANDOM_STATE)
                
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)
            logging.info(f"{self.model_type} training completed")
            return y_pred, self.y_test
        except Exception as e:
            logging.error(f"Error in training {self.model_type} model: {str(e)}")
    
    def build_deep_learning_model(self,
                                  input_dim = config.DNNConfig.INPUT_DIM,
                                  model_type = None):
        """Builds, trains, evaluates a deep learning model, and returns the classification report"""
        self.model_type = model_type
        try:
            model = Sequential([
                Dense(units=16, input_dim=input_dim, activation='relu'),
                Dense(units=24, activation='relu'),
                Dropout(0.5),
                Dense(20, activation='relu'),
                Dense(24, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=config.DNNConfig.OPTIMIZER,
                          loss=config.DNNConfig.LOSS,
                          metrics=config.DNNConfig.METRICS)
            logging.info("Deep learning model built and compiled")
            model.fit(self.X_train, self.y_train, 
                      batch_size=config.DNNConfig.BATCH_SIZE, 
                      epochs=config.DNNConfig.EPOCHS)
            y_pred = model.predict(self.X_test)
            return y_pred, self.y_test
        except Exception as e:
            logging.error(f"Error in building deep learning model: {str(e)}")
    
    def undersample_data(self):
        """Performs undersampling on the data"""
        try:
            fraud_indices = np.array(self.data[self.data == 1].index)
            normal_indices = self.data[self.data == 0].index
            random_normal_indices = np.random.choice(normal_indices, len(fraud_indices), replace=False)
            under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
            under_sample_data = self.data.iloc[under_sample_indices,:]
            X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
            y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_undersample, y_undersample, 
                                                                                    test_size=config.ModelConfig.TEST_SIZE, 
                                                                                    random_state=config.ModelConfig.RANDOM_STATE)
            logging.info("Data undersampled successfully")
        except Exception as e:
            logging.error(f"Error in undersampling data: {str(e)}")
    
    def smote_data(self):
        """Performs SMOTE on the data"""
        try:
            X = self.data.iloc[:, self.data.columns != 'Class']
            y = self.data.iloc[:, self.data.columns == 'Class']
            X_resample, y_resample = SMOTE().fit_resample(X, y.values.ravel())
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_resample, y_resample, 
                                                                                    test_size=config.ModelConfig.TEST_SIZE, 
                                                                                    random_state=config.ModelConfig.RANDOM_STATE)
            logging.info("Data oversampled using SMOTE successfully")
        except Exception as e:
            logging.error(f"Error in SMOTE: {str(e)}")
    
    def evaluate_performance(self, y_pred):
        """Evaluates the performance of the model and returns metrics"""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        return {
            "model": self.model_type,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
