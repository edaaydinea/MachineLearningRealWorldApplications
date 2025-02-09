from data_preprocessing import DataPreprocessing
from model_training import ModelTraining
import config
from config import DATA_PATH, LOGISTIC_REGRESSION_PARAMS

def train_and_evaluate(data_path, logistic_regression_params, grid_search_params):
    """Train and evaluate the model with given parameters."""
    mt = ModelTraining(data_path, logistic_regression_params, grid_search_params)
    mt.preprocess_data()
    mt.build_model()
    accuracy, precision, recall, f1 = mt.evaluate_model()
    mean_accuracy, std_accuracy = mt.cross_validate_model()
    best_accuracy, best_parameters, best_score = mt.tune_model()
    print(f"Best Accuracy: {best_accuracy},\nBest Parameters: {best_parameters},\nBest Score: {best_score}")
    return best_accuracy, best_parameters, best_score

def main():
    # Data Preprocessing
    dp = DataPreprocessing(config.DATA_PATH, config.TOP_SCREENS_PATH)
    dp.initial_cleaning()
    dp.plot_histograms()
    dp.plot_correlations()
    dp.feature_engineering()
    dp.process_screens()
    dp.create_funnels()
    dp.save_dataset(config.PROCESSED_DATA_PATH)

    # Define parameter sets
    parameter_sets = [
        (LOGISTIC_REGRESSION_PARAMS, config.GRID_SEARCH_PARAMS),
        (LOGISTIC_REGRESSION_PARAMS, config.GRID_SEARCH_PARAMS2)
    ]

    # Train and evaluate models with different parameter sets
    for logistic_regression_params, grid_search_params in parameter_sets:
        train_and_evaluate(DATA_PATH, logistic_regression_params, grid_search_params)

if __name__ == "__main__":
    main()
