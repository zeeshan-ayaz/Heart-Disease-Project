# Heart Disease Prediction
This repository contains a Jupyter Notebook for predicting heart disease using various machine learning algorithms. The dataset is sourced from the UCI Heart Disease dataset. The notebook explores the data, cleans it, removes outliers, and applies multiple algorithms to determine the best model for prediction.

# Dataset
The dataset used for this project is the UCI Heart Disease dataset. It includes several features related to heart disease diagnosis.

# Project Steps
## Exploratory Data Analysis (EDA):

Explored each column separately to understand the data distribution and relationships.
## Data Cleaning:

Created a function to remove missing values.
Removed outliers to ensure data quality.
## Modeling:

Applied various algorithms including Logistic Regression, K-Nearest Neighbors (KNN) Classifier, and Gaussian Naive Bayes.
Used an ensemble algorithm to improve model performance.
Identified Gaussian Naive Bayes as the best-performing model.
Algorithms Used
Logistic Regression: A linear model for binary classification.
KNN Classifier: A non-parametric method used for classification and regression.
Gaussian Naive Bayes: A probabilistic classifier based on Bayes' theorem.
Ensemble Algorithm: Combined multiple models to improve prediction accuracy.
Results
The Gaussian Naive Bayes algorithm provided the best results for heart disease prediction in this project.

# Usage
To run the notebook:

 #Clone the repository:
bash
## Copy code
git clone https://github.com/your-username/heart-disease-prediction.git
## Navigate to the project directory:
bash
## Copy code
cd heart-disease-prediction
## Open the Jupyter Notebook:
bash
Copy code
jupyter notebook Heart_Disease_Prediction.ipynb
Dependencies
Python 3.x
Jupyter Notebook
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Install the dependencies using:

bash
Copy code
pip install -r requirements.txt
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
UCI Machine Learning Repository for providing the dataset.
