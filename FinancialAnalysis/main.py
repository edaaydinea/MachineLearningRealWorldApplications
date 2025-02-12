# main.py
import pandas as pd
import data_preprocessing as dp
import modeling
import config
import utils


def main():
    # Data Preprocessing
    X_train, X_test, y_train, y_test, users = dp.preprocess_pipeline()
    print("Data Preprocessing complete")
    
    # Train and Compare Models
    results_df, best_model = modeling.compare_models(X_train, X_test, y_train, y_test)
    print("Model Comparison Results:")
    print(results_df)
    
    # Plot confusion matrix for the best model
    y_pred, _, _, _, _ = modeling.evaluate_performance(best_model, X_test, y_test)
    utils.plot_confusion_matrix(y_test, y_pred)
    
    # (Optional) Additional visualizations:
    # For example: Training data histograms or correlation heatmap
    # utils.plot_histograms(X_train)
    # utils.plot_correlation_heatmap(X_train)
    
    # Save final results (e.g., entry_id, actual label, and prediction) as CSV
    if users is not None:
        final_results = pd.concat([users.reset_index(drop=True),
                                   y_test.reset_index(drop=True)], axis=1)
        final_results.columns = ['entry_id', 'e_signed']
        final_results['predictions'] = y_pred
        final_results.to_csv('final_results.csv', index=False)
        print("Final results saved to final_results.csv")

if __name__ == "__main__":
    main()
