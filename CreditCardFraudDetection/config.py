import os


class DataConfig:
    BASE_DIR = "./"
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    INPUT_DATA_PATH = 'creditcard.csv'
    PROCESSED_DATA_PATH = 'processed_creditcard.csv'
    GRAPH_DIR = os.path.join(BASE_DIR, 'graphs')
    NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

class ModelConfig:
    TEST_SIZE = 0.3
    RANDOM_STATE = 0

class XgboostConfig:
    EVAL_METRIC = 'logloss'

class DNNConfig:
    INPUT_DIM = 29
    OPTIMIZER = 'adam'
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']
    EPOCHS = 5
    BATCH_SIZE = 15
    
class VisualizationConfig:
    SAVE_PLOT = True