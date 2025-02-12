import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def load_data(self, file_path = os.path.join(config.DATA_DIR, config.INPUT_DATA_PATH)):
        """
        Load data from a CSV file 
        """
        try:
            logging.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise Exception(f"Error loading data from {file_path}: {e}")

    def preprocess_data(self, df):
        """
        Clean the df and perform feature engineering:
        - Drop the 'months_employed' column
        - Create a new column 'personal_account_months' by combining 'personal_account_m' and 'personal_account_y'
        - Remove the unnecessary columns 'personal_account_m' and 'personal_account_y'
        - Applying one-hot encoding to the df
        """
        
        # Drop the 'months_employed' column
        if 'months_employed' in df.columns:
            df = df.drop(columns=['months_employed'])
            
        # Create a new column 'personal_account_months' by combining 'personal_account_m' and 'personal_account_y'
        if 'personal_account_m' in df.columns and 'personal_account_y' in df.columns:
            df['personal_account_months'] = df['personal_account_m'] + (df['personal_account_y'] * 12)
            df = df.drop(columns=['personal_account_m', 'personal_account_y'])
        
        
        # One-hot encoding
        df = pd.get_dummies(df)
        
        # Drop the 'pay_schedule_semi-monthly' column
        if 'pay_schedule_semi-monthly' in df.columns:
            df = df.drop(columns=['pay_schedule_semi-monthly'])

        return df

    def split_data(self, df, 
                   target_col = config.TARGET,
                   text_size = config.TEST_SIZE,
                   random_state = config.RANDOM_STATE):
        
        """
        Split the data into training and testing sets
        """
        
        # Save the entry_id if it exists
        users = df["entry_id"] if "entry_id" in df.columns else None
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        if 'entry_id' in X.columns:
            X = X.drop(columns=['entry_id'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = text_size, random_state = random_state)
        logging.info(f"Data split into training and testing sets: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, users

    def scale_features(self, X_train, X_test):
        """
        Scale the features using StandardScaler
        """
        logging.info("Scaling features using StandardScaler")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                      columns=X_train.columns, 
                                      index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                     columns=X_test.columns, 
                                     index=X_test.index)
        return  X_train_scaled, X_test_scaled

    def save_processed_data(self, df, file_path = os.path.join(config.DATA_DIR, config.PROCESSED_DATA_PATH)):
        try:
            logging.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")
            raise Exception(f"Error saving processed data to {file_path}: {e}")

    def preprocess_pipeline(self):
        df = self.load_data()
        df = self.preprocess_data(df)
        X_train, X_test, y_train, y_test, users = self.split_data(df)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, users

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, users = preprocessor.preprocess_pipeline()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train.head())
    print(y_train.head())
    print(X_test.head())
    print(y_test.head())
    print("Data Preprocessing complete")