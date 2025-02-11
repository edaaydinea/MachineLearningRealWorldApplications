# Configuration settings and constants

DATA_FOLDER = '../data/'
INPUT_DATA_PATH = 'churn_data.csv'
PROCESSED_DATA_PATH = 'processed_churn_data.csv'
VISUALIZATION_PATH = '../graphs/'

# Model parameters
LOGISTIC_REGRESSION_PARAMS = {
    'random_state': 0,
    'penalty': 'l2',
    'solver': 'liblinear'
}

# Grid search parameters
GRID_SEARCH_PARAMS = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

HISTOGRAM_DROP_COLS = ['user', 'churn']
PIE_CHART_COLS = ['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']
CORRELATION_DROP_COLS = ['housing', 'payment_type', 'registered_phones', 'zodiac_sign']
FEATURE_IMPORTANCE_DROP_COLS = ['housing_na', 'zodiac_sign_na', 'payment_type_na']