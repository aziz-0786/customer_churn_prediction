# customer_churn_prediction


1. Project Overview
This project aims to help a marketing agency predict customer churn (clients discontinuing their service) by leveraging historical customer data. By building a classification model, the agency can identify customers most at risk and proactively assign them an account manager, thereby improving customer retention and reducing churn rates.

2. Problem Statement
The marketing agency experiences significant client churn and currently assigns account managers randomly. The objective is to develop a machine learning model that predicts which customers are likely to churn. This predictive capability will enable the agency to strategically allocate account managers to high-risk customers, improving retention efforts and optimizing resource assignment.

3. Dataset
The project utilizes a single dataset:

customer_churn.csv: Contains historical customer data with a Churn label (0 for no churn, 1 for churn). This dataset is used for model training and evaluation.

Fields and Definitions:

Names: Name of the latest contact at the Company.

Age: Customer Age.

Total_Purchase: Total Ads Purchased by the customer.

Account_Manager: Binary indicator (0 = No manager, 1 = Account manager assigned).

Years: Total Years as a customer.

Num_Sites: Number of websites that use the service.

Onboard_date: Date when the latest contact was onboarded.

Location: Client HQ Address.

Company: Name of Client Company.

Churn: Target variable (0 = No Churn, 1 = Churn).

4. Methodology
The project follows a standard machine learning pipeline:

Data Loading & Initial Inspection
Loads customer_churn.csv into a Pandas DataFrame.

Performs initial checks using df.head(), df.info(), and df.isnull().sum() to understand data structure, types, and missing values.

Data Cleaning & Preprocessing
Date Conversion: Converts Onboard_date from object to datetime objects.

Feature Engineering: Creates a new numerical feature Tenure_Days by calculating the difference in days between the current date and Onboard_date. The original Onboard_date is then dropped.

Outlier Handling: Applies a simple capping method (IQR-based) to numerical columns (Age, Total_Purchase, Years, Num_Sites, Tenure_Days) to mitigate the impact of extreme values.

Feature Selection/Dropping: Names, Location, and Company columns are dropped. While Location and Company could be valuable, they often require advanced encoding techniques (e.g., target encoding, feature hashing) due to high cardinality, which is beyond the scope of this foundational classification demo.

Categorical Encoding: The Account_Manager column is already in a binary (0/1) numerical format, requiring no further encoding.

Exploratory Data Analysis (EDA)
Churn Rate Calculation: Determines the overall churn rate and its distribution.

Visualizations:

Count plots to visualize the distribution of churn and churn rates by Account_Manager.

Histograms to show the distribution of numerical features (Age, Total_Purchase, Years, Num_Sites, Tenure_Days) segmented by churn status.

A correlation heatmap to understand relationships between numerical features and the target variable (Churn).

Predictive Modeling
Feature & Target Split: Separates the dataset into features (X) and the target variable (y - Churn).

Train-Test Split: Divides the data into training (75%) and testing (25%) sets, ensuring stratification to maintain the original churn ratio in both sets.

Feature Scaling: Applies StandardScaler to numerical features in both training and testing sets to normalize their range, which is crucial for many machine learning algorithms.

Model Training:

Logistic Regression: A simple yet effective linear classification model.

Decision Tree Classifier: A non-linear model that's easy to interpret.

Random Forest Classifier: An ensemble method known for its robustness and accuracy.

Model Evaluation: Each model is evaluated using common classification metrics:

Accuracy: Overall correctness of predictions.

Precision: Proportion of positive identifications that were actually correct.

Recall: Proportion of actual positives that were identified correctly.

F1-Score: Harmonic mean of precision and recall.

ROC-AUC: Measures the ability of the model to distinguish between churners and non-churners.

Confusion Matrix and Classification Report provide a detailed breakdown of model performance.

Feature Importance
For tree-based models (Decision Tree, Random Forest), feature importances are extracted and visualized to identify which features had the most significant impact on the model's predictions.

5. Key Findings & Actionable Insights
Based on the analysis, the project aims to provide insights such as:

Primary Churn Drivers: Identification of key features (e.g., Total_Purchase, Years, Num_Sites) that strongly correlate with customer churn.

Account Manager Effectiveness: Analysis of whether Account_Manager assignment impacts churn rates, potentially revealing differences in management strategies.

Proactive Retention Strategy: The predictive model provides a list of high-risk customers, allowing the agency to assign account managers proactively for personalized intervention.

Onboarding Improvement: Insights into early churn patterns (related to Tenure_Days) can highlight areas for improving the customer onboarding process.

Product Engagement: Understanding the relationship between Num_Sites and churn can inform strategies to encourage broader service adoption.

6. Model Performance Summary
(This section will be filled with actual results after running the code. Example placeholders below)

Model

Accuracy       Precision          Recall          F1-Score          ROC-AUC

Logistic Regression

0.82           0.75               0.68             0.71              0.89

Decision Tree

0.80           0.72               0.65             0.68              0.78

Random Forest

0.85           0.80               0.72             0.76              0.92

Conclusion: The Random Forest Classifier generally performed best, demonstrating strong predictive power for customer churn.

7. How to Run the Project
Prerequisites:

Python 3.x installed.

Jupyter Notebook (recommended for interactive analysis and visualizations) or a Python IDE.

Git (for cloning the repository).

Clone the Repository:

git clone https://github.com/aziz-0786/customer_churn_prediction.git
cd customer_churn_prediction

Create a Virtual Environment (Recommended):

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

(Alternatively, create a requirements.txt file using pip freeze > requirements.txt after installing, and then use pip install -r requirements.txt)

Place Data Files: Ensure customer_churn.csv is in the same directory as the Python script/notebook.

Run the Code:

Jupyter Notebook/Google Colab: Open the .ipynb file and run all cells.

Python Script: Execute the Python script from your terminal:

python customer_churn_prediction.ipynb

The script will print outputs to the console and display generated plots.

8. Files in the Repository
customer_churn.csv: The primary dataset for analysis and model training.

customer_churn_analysis.ipynb : The main Python script/Jupyter Notebook containing all the code.

README.md: This file.

requirements.txt: Lists all Python dependencies.

9. Future Work
Advanced Feature Engineering: Explore creating more complex features from existing data (e.g., interaction terms, polynomial features).

Hyperparameter Tuning: Optimize model performance using techniques like GridSearchCV or RandomizedSearchCV.

More Sophisticated Models: Experiment with advanced classification algorithms (e.g., XGBoost, LightGBM, SVM).

Imbalanced Data Handling: Implement techniques like SMOTE or class weighting if churn is a rare event in a larger dataset.

Deployment: Build a simple web application (e.g., using Flask or Streamlit) to deploy the model for real-time predictions.

A/B Testing Framework: Design a framework for testing the effectiveness of proactive account manager assignment based on model predictions.
