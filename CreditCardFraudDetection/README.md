# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.

## Project Structure

```plaintext
CreditCardFraudDetection/
│
├── data/
│   ├── creditcard.csv
│   ├── processed_creditcard.csv
│   └── results.csv
│
├── graphs/
│   └── *.png
│
├── notebooks/
│   └── *.ipynb
│
├── CreditCardFraudDetection/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── main.py
│   ├── modeling.py
│   └── utils.py
│
└── README.md
```

## Installation

1. Clone the repository:

    ```sh
    git clone git@github.com:edaaydinea/MachineLearningRealWorldApplications.git
    cd MachineLearningRealWorldApplications
    cd CreditCardFraudDetection
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```


## Usage

### Run the Pipeline

To run the pipeline, execute the following command:

```sh
   python main.py
```


### Configuration

The `config.py` file contains various configuration settings for the project, including paths to data files, model parameters, and visualization settings.

### Results

- The results of the model evaluations are saved in `data/results.csv`.
- Confusion matrix plots for each model are saved in the `graphs/` directory.

## Project Workflow

1. **Data Loading and Preprocessing**:
    - The `DataPreprocessor` class in `data_preprocessing.py` handles loading and preprocessing of the data.
    - The data is normalized and saved as `processed_creditcard.csv`.

2. **Model Training and Evaluation**:
    - The `ModelTrainer` class in `modeling.py` handles the training and evaluation of various machine learning models.
    - Models include Decision Tree, Random Forest, XGBoost, LightGBM, and a Deep Learning model.
    - The `evaluate_performance` method calculates accuracy, precision, recall, and F1 score for each model.
    - Confusion matrices are plotted and optionally saved using the `ConfusionMatrixPlotter` class in `utils.py`.

3. **Results**:
    - The results of the model evaluations are saved to `data/results.csv`.
    - Confusion matrix plots are saved in the `graphs/` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
