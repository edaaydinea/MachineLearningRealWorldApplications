import os

# Folder information
BASE_DIR = "../"
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DATA_PATH = 'financial_data.csv'
PROCESSED_DATA_PATH = 'processed_financial_data.csv'
GRAPH_DIR = os.path.join(BASE_DIR, 'graphs')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# DATA PARAMETERS
TARGET = 'e_signed'

# Model parameters
RANDOM_STATE = 0
TEST_SIZE = 0.2

# Logistical Regression parameters
PENALTY = 'l1'
SOLVER = 'liblinear'

# Random Forest parameters
N_ESTIMATORS = 100
CRITERION = 'entropy'