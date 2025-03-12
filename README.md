# **Wine Quality Prediction Using Machine Learning**  

## **Overview**  
This project aims to predict the quality of red wine based on various physicochemical properties using machine learning techniques. It involves data collection, preprocessing, exploratory data analysis, feature engineering, and model training using different classifiers.  

---

## **Dataset**  
The dataset used for this project is the **Red Wine Quality Dataset** from the UCI Machine Learning Repository. It consists of **1,599 samples** and **12 features**, including:  

- **Fixed acidity**  
- **Volatile acidity**  
- **Citric acid**  
- **Residual sugar**  
- **Chlorides**  
- **Free sulfur dioxide**  
- **Total sulfur dioxide**  
- **Density**  
- **pH**  
- **Sulphates**  
- **Alcohol**  
- **Quality (target variable: 0 or 1)**  

The target variable has been transformed into a **binary classification**:  
- **Good quality (1) → Quality >= 7**  
- **Poor quality (0) → Quality < 7**  

---

## **Dependencies**  
The following Python libraries are used:  

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import stats
```

Install dependencies using:  
```bash
pip install numpy pandas matplotlib seaborn imbalanced-learn scikit-learn xgboost
```

---

## **Project Workflow**  

### **1. Data Collection & Preprocessing**  
- Load the dataset into a Pandas DataFrame  
- Check for missing values and handle them  
- Normalize/scale the dataset if necessary  

### **2. Exploratory Data Analysis (EDA)**  
- **Descriptive statistics** (mean, std, min, max)  
- **Visualizations using Seaborn & Matplotlib**  
  - Count plots for wine quality distribution  
  - Bar plots for correlation between features and wine quality  
  - Heatmaps for feature correlation  

### **3. Feature Engineering**  
- Removed the target column from features  
- Transformed the quality label into a binary classification  

### **4. Data Splitting & Oversampling**  
- Split the dataset into **training (80%)** and **testing (20%)**  
- Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset  

### **5. Model Selection & Training**  
- Implemented and evaluated multiple models:  
  - **Logistic Regression**  
  - **Random Forest Classifier**  
  - **Gradient Boosting Classifier**  
  - **Support Vector Machine (SVM)**  
  - **XGBoost Classifier**  
- Used **GridSearchCV** for hyperparameter tuning  

### **6. Model Evaluation**  
- **Accuracy Score**  
- **Mean Squared Error (MSE)**  
- **Confusion Matrix & Classification Report**  

---

## **Results**  
The models were evaluated based on accuracy, and the **best-performing model** was:  
- **XGBoost Classifier** with **accuracy > 85%**  

---

## **How to Run the Project**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/wine-quality-prediction.git
   cd wine-quality-prediction
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script:  
   ```bash
   python wine_quality_prediction.py
   ```

---

## **Future Enhancements**  
- Implement deep learning models (ANNs)  
- Add more feature selection techniques  
- Deploy the model as a web app using **Flask/Streamlit**  

---

## **Contributors**  
- **Shalu Yadav** (Developer & Data Analyst)  

---
