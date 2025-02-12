import pandas as pd
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Modeling:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.info("Modeling class initialized")

    def evaluate_performance(self, model, X_test, y_test):
        """
        Evaluate the performance of a model
        """
        logging.info("Evaluating model performance")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return y_pred, acc, prec, rec, f1

    def train_logistic_regression(self, X_train, y_train):
        """
        Train a logistic regression model
        """
        logging.info("Training Logistic Regression model")
        clf = LogisticRegression(random_state=config.RANDOM_STATE,
                                 penalty=config.PENALTY,
                                 solver=config.SOLVER)
        clf.fit(X_train, y_train)
        return clf

    def train_svm_linear(self, X_train, y_train):
        """
        Train a SVM model with a linear kernel
        """
        logging.info("Training SVM model with linear kernel")
        clf = SVC(random_state=config.RANDOM_STATE,
                  kernel='linear',
                  probability=True)
        clf.fit(X_train, y_train)
        return clf

    def train_svm_rbf(self, X_train, y_train):
        """
        Train a SVM model with a rbf kernel
        """
        logging.info("Training SVM model with RBF kernel")
        clf = SVC(random_state=config.RANDOM_STATE,
                  kernel='rbf',
                  probability=True)
        clf.fit(X_train, y_train)
        return clf

    def train_random_forest(self, X_train, y_train):
        """
        Train a Random Forest model
        """
        logging.info("Training Random Forest model")
        classifier = RandomForestClassifier(random_state=config.RANDOM_STATE, 
                                            n_estimators=config.N_ESTIMATORS, 
                                            criterion=config.CRITERION)
        classifier.fit(X_train, y_train)
        return classifier

    def grid_search_rf(self, X_train, y_train, criterion='entropy'):
        """
        Applies a two-stage GridSearchCV for Random Forest.
        Uses a wide parameter range in the first stage, and a narrowed range in the second stage.
        """
        logging.info(f"Starting grid search for Random Forest with criterion: {criterion}")
        if criterion == 'entropy':
            parameters = {
                "max_depth": [3, None],
                "max_features": [1, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
                "bootstrap": [True, False],
                "criterion": ["entropy"]
            }
        else:  # 'gini'
            parameters = {
                "max_depth": [3, None],
                "max_features": [1, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
                "bootstrap": [True, False],
                "criterion": ["gini"]
            }
        rf = RandomForestClassifier(random_state=config.RANDOM_STATE, n_estimators=config.N_ESTIMATORS)
        grid_search = GridSearchCV(estimator=rf,
                                   param_grid=parameters,
                                   scoring="accuracy",
                                   cv=10,
                                   n_jobs=-1)
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        duration = time.time() - start_time
        logging.info(f"Grid search with {criterion} took {duration:.2f} seconds")
        
        # Second stage : Reducing the range of parameters
        if criterion == 'entropy':
            parameters_refined = {
                "max_depth": [None],
                "max_features": [3, 5, 7],
                "min_samples_split": [8, 10, 12],
                "min_samples_leaf": [1, 2, 3],
                "bootstrap": [True],
                "criterion": ["entropy"]
            }
        else:
            parameters_refined = {
                "max_depth": [None],
                "max_features": [8, 10, 12],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [8, 10, 12],
                "bootstrap": [True],
                "criterion": ["gini"]
            }
        grid_search_refined = GridSearchCV(estimator=rf,
                                           param_grid=parameters_refined,
                                           scoring="accuracy",
                                           cv=10,
                                           n_jobs=-1)
        start_time = time.time()
        grid_search_refined.fit(X_train, y_train)
        duration = time.time() - start_time
        logging.info(f"Refined grid search with {criterion} took {duration:.2f} seconds")
        
        best_model = grid_search_refined.best_estimator_
        best_params = grid_search_refined.best_params_
        best_score = grid_search_refined.best_score_
        logging.info(f"Best params ({criterion}): {best_params} with score: {best_score:.4f}")
        
        return best_model, best_params, best_score

    def compare_models(self, X_train, X_test, y_train, y_test):
        """
        Train different models and compare their performance on the test set.
        The results are collected in a DataFrame.
        """
        logging.info("Comparing different models")
        results = []
        
        # Logistic Regression
        model_lr = self.train_logistic_regression(X_train, y_train)
        _, acc, prec, rec, f1 = self.evaluate_performance(model_lr, X_test, y_test)
        results.append({
            'Model': 'Logistic Regression (L1)',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # SVM (Linear)
        model_svm_lin = self.train_svm_linear(X_train, y_train)
        _, acc, prec, rec, f1 = self.evaluate_performance(model_svm_lin, X_test, y_test)
        results.append({
            'Model': 'SVM (Linear)',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # SVM (RBF)
        model_svm_rbf = self.train_svm_rbf(X_train, y_train)
        _, acc, prec, rec, f1 = self.evaluate_performance(model_svm_rbf, X_test, y_test)
        results.append({
            'Model': 'SVM (RBF)',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # Random Forest (default)
        model_rf = self.train_random_forest(X_train, y_train)
        _, acc, prec, rec, f1 = self.evaluate_performance(model_rf, X_test, y_test)
        results.append({
            'Model': 'Random Forest (n=100)',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # Random Forest Grid Search (Entropy)
        model_rf_gs_entropy, params_entropy, score_entropy = self.grid_search_rf(X_train, y_train, criterion='entropy')
        _, acc, prec, rec, f1 = self.evaluate_performance(model_rf_gs_entropy, X_test, y_test)
        results.append({
            'Model': 'Random Forest (n=100, GSx2 + Entropy)',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # Random Forest Grid Search (Gini)
        model_rf_gs_gini, params_gini, score_gini = self.grid_search_rf(X_train, y_train, criterion='gini')
        _, acc, prec, rec, f1 = self.evaluate_performance(model_rf_gs_gini, X_test, y_test)
        results.append({
            'Model': 'Random Forest (n=100, GSx2 + Gini)',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        results_df = pd.DataFrame(results)
        return results_df, model_rf_gs_gini  # Returning the best model (for example, the last one)


if __name__ == "__main__":
    modeling = Modeling()
    logging.info("Modeling started")
