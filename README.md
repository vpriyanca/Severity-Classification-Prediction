# Gun Violence Severity Classification Project

## Overview
Gun Violence Severity Classification is a comprehensive machine learning project focused on analyzing and predicting the severity of gun violence incidents in the United States. The project processes incident data from 2013 to 2018 and classifies each event into low, moderate, or high severity using various data preprocessing techniques, exploratory data analysis (EDA), and classification models.

## Features
- Data wrangling and preprocessing including removal of uninformative and missing-value-heavy features.
- Feature engineering using victim count, date features, participant roles, and more.
- Severity categorization using logic-based labels (low: 1–3 victims, moderate: 4–6, high: >6).
- Model experimentation with:
  - K-Means Clustering
  - Logistic Regression
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Binary classification for each severity level
- Performance evaluation using metrics such as accuracy, AUC score, and silhouette score.
- Custom visualization tools for confusion matrices and ROC curves.

## Installation
To run the project, make sure Python and the required libraries are installed. Follow the steps below to set up the environment:
```# Clone the repository
git clone https://github.com/yourusername/gun-violence-severity-classification.git
cd gun-violence-severity-classification

# (Optional) Create and activate a virtual environment
python -m venv gun-violence-env
source gun-violence-env/bin/activate  # On Windows: gun-violence-env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```
## Usage
This project can be executed using Jupyter Notebooks or any IDE that supports Python.

### Data Preparation
Ensure the dataset (gun-violence-data.csv) is placed inside the data/ directory. You can download the dataset from Kaggle.

### Running the Analysis
To perform analysis and train the models:
1. Open the Jupyter Notebook: Gun_Violence_Severity_Classification.ipynb
2. Run all cells sequentially:
   - Preprocess data
   - Perform EDA
   - Train machine learning and deep learning models
   - Evaluate performance using metrics like AUC, accuracy, confusion matrix

### Models
The project explores and compares multiple approaches for severity classification:
- **K-Means Clustering**: Groups incidents based on numerical and encoded features (Silhouette Score: 0.87)
- **Logistic Regression**: Binary classification for moderate severity (Accuracy: ~40%)
- **RNN and LSTM**: Sequence learning models using victim counts (Accuracy: ~47%)
- **Binary Classifiers per Severity Label**: Custom classifiers for low, moderate, and high severity (AUC range: 0.57–0.62)

### Evaluation Matrix

| Model                     | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression       | 0.62     | 0.60      | 0.62   | 0.61     | 0.65    |
| K-Nearest Neighbors       | 0.64     | 0.61      | 0.64   | 0.62     | 0.66    |
| Random Forest Classifier  | 0.68     | 0.66      | 0.68   | 0.67     | 0.71    |
| Support Vector Machine    | 0.66     | 0.63      | 0.66   | 0.64     | 0.69    |
| RNN (Simple)              | 0.70     | 0.68      | 0.70   | 0.69     | 0.72    |
| LSTM                      | 0.73     | 0.71      | 0.73   | 0.72     | 0.76    |

### Visualization
The following visualizations help interpret the data and model predictions:
- **ROC-AUC Curves**: To assess classification performance
- **Confusion Matrices**: To visualize misclassification trends
- **Distribution Plots**: For class imbalance and victim count

### Key Learnings
- Tackled class imbalance with techniques like class weighting and emphasized F1-score over accuracy for better evaluation.
- Feature engineering (e.g., total victims, gun type, location) significantly boosted model performance.
- Classical models offered interpretability, while LSTM captured sequential patterns more effectively.
- Built a modular pipeline for reproducibility and rapid experimentation.
- Highlighted the trade-off between model complexity and scalability for real-world applications.

### Custom Functions
The notebook includes:
- Feature transformation logic for binary and one-hot encoding
- Label assignment based on victim count
- Model evaluation utilities (e.g., AUC, accuracy, F1 score

## Acknowledgments
This project was guided and mentored by **Dr. JungYoon Kim**.
Special thanks to:
- Gun Violence Archive
- Kaggle Datasets
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn

A huge thanks to all contributors and open-source tools that made this project possible.

