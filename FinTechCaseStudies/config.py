# Configuration settings and constants

DATA_PATH = 'appdata10.csv'
TOP_SCREENS_PATH = 'top_screens.csv'
PROCESSED_DATA_PATH = 'new_appdata10.csv'

# Model parameters
LOGISTIC_REGRESSION_PARAMS = {
    'random_state': 0,
    'penalty': 'l1',
    'solver': 'liblinear'  # Update solver to 'liblinear' which supports 'l1' penalty
}

# Grid search parameters
GRID_SEARCH_PARAMS = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

GRID_SEARCH_PARAMS2 = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 0.5, 0.9, 1, 2, 5]
}
