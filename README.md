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

### How to run

### Step-by-Step Instructions to Run the Code

#### Prerequisites

* **Software:**
  * Python 3.x
  * Jupyter Notebook or Jupyter Lab
  * Kaggle API (if you want to download the data directly from Kaggle)
* **Hardware (Recommended for "Optimized Solution.ipynb"):**
  * A system with a GPU is highly recommended due to the use of GPU-accelerated libraries (PyTorch, XGBoost with `gpu_hist`, CatBoost with `task_type: 'GPU'`).

#### **Step 1: Set Up Kaggle API**

1. **Create a Kaggle Account:** If you don't have one, sign up at [https://www.kaggle.com/](https://www.kaggle.com/).
2. **Generate API Token:**
  * Go to your Kaggle account page.
  * Click on "Create New API Token".
  * This will download a file named `kaggle.json`.
3. **Place `kaggle.json`:**
  * Move this file to the directory `~/.kaggle/` on Linux/macOS or `C:\\Users\\<username>\\.kaggle\\` on Windows.
  * If the `.kaggle` directory doesn't exist, create it.

#### **Step 2: Clone the Repository**

1. Open your terminal or command prompt.
  
2. Clone this repository to your local machine using Git:
  
      git clone https://github.com/Zues1364/Child-Mind-Institute-Problematic-Internet-Use-From-Chill-Guys.git
   
      cd Child-Mind-Institute-Problematic-Internet-Use-From-Chill-Guys
  

#### **Step 3: Install Required Libraries**

1. Navigate to the cloned repository directory:
  
2. It's recommended to create a virtual environment to avoid conflicts with other Python projects:
  
  * **Using `venv` (Recommended):**
    
        python3 -m venv .venv
        source .venv/bin/activate  # On Linux/macOS
        .venv\\Scripts\\activate  # On Windows
    
3. Install the required libraries using `pip`:
  
      pip install -r requirements.txt
   
  
  #### **Step 4: Download the Dataset**
  
4. **Using Kaggle API:**
  
  * Ensure you have joined the "Child Mind Institute - Problematic Internet Use" competition on Kaggle.
    
  * If your **notebook** is configured to run on a Kaggle environment:
    
    * The dataset is likely already available in the `/kaggle/input/` directory.
  * If you are running **locally**:
    
    * Use the Kaggle API to download the dataset:
      
          kaggle competitions download -c child-mind-institute-problematic-internet-use
      
    * Unzip the downloaded data:
      
          unzip child-mind-institute-problematic-internet-use.zip -d input
      
    * **Or** you can manually download and unzip into the `input` folder.
      

#### **Step 5: Run the Jupyter Notebooks**

1. **Start Jupyter Notebook or Jupyter Lab:**
  
      jupyter notebook
  
  or
  
      jupyter lab
  
2. **Open the Notebooks:**
  
  * Navigate to the directory where you cloned the repository.
  * Open either **First Solution to CMI.ipynb** or **Optimized Solution.ipynb**.
3. **Run the Cells:**
  
  * Execute each cell in the notebook sequentially by pressing `Shift + Enter`.
  * Make sure to read the comments and understand what each cell does.

**Specific Considerations for Each Notebook:**

* **First Solution to CMI.ipynb:**
  
  * This notebook can be run on a CPU, although a GPU will speed up the LightGBM training.
  * Ensure that the paths to the data files are correct.
  * After running the notebook, the submission file will be generated as `submission.csv`.
* **Optimized Solution.ipynb:**
  
  * **GPU Usage:** This notebook is designed to utilize a GPU. If you don't have a GPU, you will need to modify the following:
    * Change `device: 'cpu'` in `Params` for LightGBM.
    * Change `tree_method: 'hist'` (or remove the line) in `XGB_Params` for XGBoost.
    * Change `task_type: 'CPU'` in `CatBoost_Params` for CatBoost.
  * **Autoencoder:** Pay close attention to the AutoEncoder section. The number of epochs and batch size might need adjustment based on your hardware.
  * **KNNImputer:** This step can be computationally intensive. If you have limited RAM, you might need to reduce the dataset size or use a more memory-efficient imputation method.

#### **Step 6: Submit Predictions (Optional)**

1. **Upload to Kaggle:**
  * Go to the "Child Mind Institute - Problematic Internet Use" competition page on Kaggle.
  * Click on "Submit Predictions".
  * Upload the generated `submission.csv` file.

---

## Notes
- The "First Solution" is a good starting point to understand the dataset and build a basic model.
- The "Optimized Solution" incorporates advanced techniques for higher accuracy and is recommended for competitive performance.

---

## Acknowledgments
These solutions leverage domain knowledge, feature engineering, and ensemble modeling to tackle the complex problem of predicting problematic internet usage. Feedback and suggestions are welcome to further refine the approaches.

