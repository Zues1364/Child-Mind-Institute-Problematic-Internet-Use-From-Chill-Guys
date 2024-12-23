# Child-Mind-Institute-Problematic-Internet-Use-From-Chill-Guys

## Overview
This repository contains two Jupyter notebooks that provide solutions for the "Child Mind Institute - Problematic Internet Use" competition hosted on Kaggle. The objective of the competition is to predict problematic internet usage levels using physical activity and other associated features.

### Files:
1. **First Solution to CMI.ipynb**
2. **Optimized Solution.ipynb**

---

## Solution Details

### **First Solution to CMI.ipynb**

#### **Objective:**
This solution focuses on exploring the dataset, performing feature engineering, and training an initial model to establish a baseline performance.

#### **Key Steps:**
1. **Data Preparation:**
   - Combined tabular data with time-series features using statistical summaries.
   - Handled missing values and categorical variables through encoding techniques.
   - Feature selection included domain-specific insights and time-series statistical properties.

2. **Feature Engineering:**
   - Added interaction terms and transformations based on domain knowledge.
   - Processed categorical variables into numerical formats using mapping techniques.

3. **Modeling Approach:**
   - Utilized ensemble models like LightGBM, XGBoost, and CatBoost.
   - Employed cross-validation with stratified splits to ensure robust performance.

4. **Evaluation:**
   - Used Quadratic Weighted Kappa (QWK) as the primary evaluation metric.

   **Quadratic Weighted Kappa Formula:**
   
$\text{QWK} = 1 - \frac{\sum_{i=1}^n \sum_{j=1}^m \omega[i][j] \cdot O[i][j]}{\sum_{i=1}^n \sum_{j=1}^m \omega[i][j] \cdot E[i][j]}$





   Where:
   - $\omega[i][j]$ is the weight matrix.
   - $O[i][j]$ is the observed frequency matrix.
   - $E[i][j]$ is the expected frequency matrix.

   - Generated submission files using the best-performing model.

---

### **Optimized Solution.ipynb**

#### **Objective:**
This notebook refines the baseline solution by incorporating advanced modeling techniques and deeper feature extraction to improve performance.

#### **Key Steps:**
1. **Advanced Feature Engineering:**
   - Implemented additional derived features such as body composition ratios and combined metrics.
   - Dropped less relevant features based on domain relevance.

2. **AutoEncoder for Time-Series Features:**
   - Employed an autoencoder neural network to reduce dimensionality of time-series data.

   **AutoEncoder Loss Function:**
   $\text{Loss} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2$

   Where:
   - $x_i$ is the original input.
   - $x_hat_i$ is the reconstructed output.

   - Encoded features were merged back into the dataset for training.

3. **Missing Data Handling:**
   - Applied KNN imputation to fill missing numerical values effectively.

4. **Modeling Approach:**
   - Utilized a Voting Regressor combining LightGBM, XGBoost, and CatBoost for improved predictions.
   - Fine-tuned thresholds using an optimization routine to maximize QWK.

   **Threshold Optimization Formula:**
   Optimized Predictions =
   - 0, if $y_hat$ < $theta_1$
   - 1, if $theta_1$ <= $y_hat$ < $theta_2$
   - 2, if $theta_2$ <= $y_hat$ < $theta_3$
   - 3, otherwise

   Where:
   - $y_hat$ is the model's raw prediction.
   - $theta_1$, $theta_2$, $theta_3$ are optimized thresholds.

5. **Evaluation:**
   - Optimized final predictions using thresholding techniques.
   - Provided feature importance analysis to understand key contributors to the model.

---

## Usage
1. Ensure you have the required Python libraries installed, including `numpy`, `pandas`, `torch`, `scikit-learn`, `optuna`, and `lightgbm`.
2. Download the dataset from the Kaggle competition page and place it in the appropriate input directory.
3. Run the notebooks sequentially to replicate the solutions and generate predictions.

---

## Notes
- The "First Solution" is a good starting point to understand the dataset and build a basic model.
- The "Optimized Solution" incorporates advanced techniques for higher accuracy and is recommended for competitive performance.

---

## Acknowledgments
These solutions leverage domain knowledge, feature engineering, and ensemble modeling to tackle the complex problem of predicting problematic internet usage. Feedback and suggestions are welcome to further refine the approaches.

