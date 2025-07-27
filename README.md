# Hospital Survival Rate Prediction

## Project Overview

This project focuses on predicting hospital survival rates using machine learning techniques. The goal is to develop a model that can accurately predict whether a patient will survive their hospital stay based on various clinical and demographic features.

## 📊 Dataset Description

The project uses a comprehensive medical dataset containing patient information with the following characteristics:

- **Target Variable**: `hospital_death` (binary: 0 = survived, 1 = died)
- **Dataset Size**: Large-scale medical dataset with multiple features
- **Data Types**: Mixed (numerical, categorical, and boolean after encoding)
- **Data Quality**: Contains missing values requiring preprocessing

### Data Files

- `dataset.csv` - Original dataset (30MB)
- `df_original.csv` - Processed original data (40MB)
- `df_dropped_values.csv` - Dataset with missing values dropped (28MB)
- `df_imputated_values.csv` - Dataset with missing values imputed (47MB)

## 🏗️ Project Structure

```
Survival-Rate-Prediction/
├── EDApart.ipynb              # Exploratory Data Analysis
├── model_1.ipynb              # Machine Learning Model Implementation
├── dataset.csv                # Original dataset
├── df_original.csv            # Processed original data
├── df_dropped_values.csv      # Data with dropped missing values
├── df_imputated_values.csv    # Data with imputed missing values
├── Figures/                   # Generated visualizations
│   └── null value distribution.png
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🔍 Methodology

### 1. Data Preprocessing

The project implements two different approaches for handling missing values:

1. **Dropping Strategy**: Remove rows with missing values (`df_dropped_values.csv`)
2. **Imputation Strategy**: Fill missing values with appropriate methods (`df_imputated_values.csv`)

### 2. Feature Engineering

- **Data Type Conversion**: Boolean columns converted to integers for model compatibility
- **Scaling**: RobustScaler applied to handle outliers effectively
- **Normalization**: Normalizer applied to scale features to unit norm
- **Value Clipping**: Extreme values clipped to prevent overflow issues

### 3. Model Implementation

The project implements **Logistic Regression** with the following configurations:

#### Model 1 (Dropped Values Dataset)
```python
LogisticRegression(
    solver="liblinear",
    penalty="l2",
    C=0.05,
    max_iter=10000,
    tol=1e-3
)
```

#### Model 2 (Imputed Values Dataset)
```python
LogisticRegression(
    solver='saga',
    max_iter=500,
    C=1.0,
    penalty='l2'
)
```

## 📈 Results

### Model Performance

| Dataset Strategy | Training Accuracy | Test Accuracy |
|------------------|-------------------|---------------|
| Dropped Values   | 91.81%           | 91.95%        |
| Imputed Values   | 92.09%           | 92.94%        |

### Key Findings

- **Imputation Strategy Performs Better**: The model trained on imputed data shows higher accuracy
- **Consistent Performance**: Both models show similar training and test accuracy, indicating good generalization
- **Robust Scaling**: The use of RobustScaler helps handle outliers effectively

## 🛠️ Installation & Setup

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Required Python Packages

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization

## 📖 Usage

### Running the Analysis

1. **Exploratory Data Analysis**:
   ```bash
   jupyter notebook EDApart.ipynb
   ```

2. **Model Training and Evaluation**:
   ```bash
   jupyter notebook model_1.ipynb
   ```

### Data Processing Pipeline

1. Load the original dataset
2. Handle missing values (drop or impute)
3. Convert boolean columns to integers
4. Apply RobustScaler for outlier handling
5. Apply Normalizer for feature scaling
6. Split data into training and test sets
7. Train Logistic Regression models
8. Evaluate model performance

## 🔬 Technical Details

### Data Preprocessing Steps

1. **Missing Value Handling**:
   - Drop strategy: Remove rows with any missing values
   - Imputation strategy: Fill missing values using appropriate methods

2. **Feature Scaling**:
   - RobustScaler: Handles outliers better than StandardScaler
   - Value clipping: Prevents overflow with extreme values
   - Normalizer: Scales features to unit norm

3. **Model Configuration**:
   - L2 regularization to prevent overfitting
   - Different solvers (liblinear, saga) for optimization
   - Hyperparameter tuning for regularization strength

### Model Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Classification Report**: Detailed precision, recall, and F1-score
- **Cross-validation**: Ensures robust performance estimation

## 📊 Visualizations

The project includes various visualizations in the `Figures/` directory:

- **Null Value Distribution**: Heatmap showing missing data patterns
- **Categorical Variable Analysis**: Count plots for categorical features
- **Numerical Variable Analysis**: Histograms for numerical features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔮 Future Improvements

- Implement ensemble methods (Random Forest, XGBoost)
- Add feature selection techniques
- Implement cross-validation for more robust evaluation
- Add model interpretability analysis
- Deploy model as a web application
- Add real-time prediction capabilities

---

**Note**: This project uses a sample of 25,000 records for model training due to computational constraints. For production use, consider using the full dataset with appropriate computational resources. 