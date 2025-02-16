import logging
import pandas as pd
from sklearn.metrics import confusion_matrix
from data_preprocessing import DataPreprocessor
from modeling import ModelTrainer
from utils import ConfusionMatrixPlotter
import config
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(y_test, y_pred, model_type):
    cm_matrix = confusion_matrix(y_test, y_pred.round())
    cm_plotter = ConfusionMatrixPlotter(cm_matrix, classes=['Not Fraud', 'Fraud'])
    cm_plotter.plot(save_plot=config.VisualizationConfig.SAVE_PLOT, model_type=model_type)

def train_and_evaluate(model_type, results):
    trainer = ModelTrainer(model_type)
    trainer.data_splitting()
    y_pred, y_test = trainer.training()
    results.append(trainer.evaluate_performance(y_pred))
    plot_confusion_matrix(y_test, y_pred, model_type)
    
    
def evaluate_deep_learning_model(results, dnn_trainer, model_type):
    y_pred, y_test = dnn_trainer.build_deep_learning_model(model_type= model_type)
    results.append(dnn_trainer.evaluate_performance(y_pred.round()))
    plot_confusion_matrix(y_test, y_pred, model_type)

def main():
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.preprocess()
    
    results = []

    # Train and evaluate traditional models
    for model in ["decision_tree", "random_forest", "xgboost", "lightgbm"]:
        logging.info(f"Training {model}...")
        train_and_evaluate(model, results)
        
    dnn_trainer = ModelTrainer()
    
    # Train and evaluate Deep Learning Model without undersampling
    logging.info("Training Deep Learning Model without undersampling...")
    dnn_trainer.data_splitting()
    evaluate_deep_learning_model(results, dnn_trainer, model_type="dnn")
    
    # With undersampling
    logging.info("Training Deep Learning Model with undersampling...")
    dnn_trainer.undersample_data()
    evaluate_deep_learning_model(results, dnn_trainer, model_type="dnn_undersampled")
    
    # With SMOTE
    logging.info("Training Deep Learning Model with SMOTE...")
    dnn_trainer.smote_data()
    evaluate_deep_learning_model(results, dnn_trainer, model_type="dnn_smote")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config.DataConfig.DATA_DIR, 'results.csv'), index=False)
    logging.info("Results saved to results.csv")


if __name__ == "__main__":
    main()
