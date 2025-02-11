from data_preprocessing import DataPreprocessing
from model_training import ModelTraining
import config
from config import DATA_PATH, LOGISTIC_REGRESSION_PARAMS, GRID_SEARCH_PARAMS
from utils import plot_histograms, plot_pie_charts, plot_correlation_matrix

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
    dp = DataPreprocessing(config.DATA_PATH)
    dp.initial_cleaning()
    
    # Plot histograms
    plot_histograms(dp.dataset)
    
    # Plot pie charts
    plot_pie_charts(dp.dataset)
    
    # Plot correlation matrix
    plot_correlation_matrix(dp.dataset)
    dp.save_dataset(config.PROCESSED_DATA_PATH)
    

    # Train and evaluate model
    train_and_evaluate(config.PROCESSED_DATA_PATH, LOGISTIC_REGRESSION_PARAMS, GRID_SEARCH_PARAMS)

if __name__ == "__main__":
    main()
