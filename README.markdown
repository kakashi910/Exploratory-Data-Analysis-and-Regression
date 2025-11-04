# Exploratory Data Analysis, Regression, and Classification Project

## Overview
This project performs Exploratory Data Analysis (EDA), regression, and classification on two datasets: the **House Prices Dataset** (from Kaggle) and the **Titanic Dataset** (from a public CSV). The objective is to predict house prices using Linear Regression and passenger survival using Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) models. The project includes data visualization, feature engineering, model training, and evaluation.

## Datasets
- **House Prices Dataset**: Sourced from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). Contains 1460 samples with 81 features (e.g., LotArea, OverallQual, Neighborhood) and target variable `SalePrice`.
  - **Note**: Download `train.csv` from the Kaggle link and place it in the working directory.
- **Titanic Dataset**: Sourced from [GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv). Contains 891 samples with features like Age, Sex, Pclass, and target variable `Survived` (binary: 0=did not survive, 1=survived).

## Prerequisites
- **Python 3.x** environment (e.g., Anaconda or virtualenv).
- **Required Libraries**:
  - Install via pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
- **Jupyter Notebook**: Install via `pip install notebook` or use Anaconda.
- **Dataset**:
  - Download `train.csv` from the House Prices Kaggle page and place it in the working directory.
  - Titanic dataset is loaded directly from the provided URL.

## Instructions
1. **Setup Environment**:
   - Ensure Python and required libraries are installed.
   - Start a Jupyter Notebook server: `jupyter notebook`.

2. **Run the Notebook**:
   - Open `project_notebook.ipynb` in Jupyter.
   - Run all cells sequentially to:
     - Load and preprocess datasets.
     - Perform EDA (pairplots, heatmaps).
     - Engineer features (handle missing values, encode categoricals).
     - Train and evaluate Linear Regression (House Prices) and Logistic Regression, SVM, KNN (Titanic).
     - Visualize results (regression plots, confusion matrices).

3. **Export Outputs**:
   - Export the notebook to PDF: In Jupyter, go to `File > Download as > PDF via LaTeX` (requires pandoc and LaTeX).
   - Create presentation slides (4-5 slides) using outputs (e.g., heatmaps, regression plots, confusion matrices) in PowerPoint or Google Slides.

4. **Key Outputs**:
   - EDA: Correlation heatmaps, pairplots showing relationships (e.g., SalePrice ~ OverallQual, Survival ~ Pclass).
   - Models:
     - Linear Regression: Mean Squared Error (MSE) for house price prediction.
     - Logistic Regression, SVM, KNN: Accuracy and confusion matrices for survival prediction.

## Project Steps
1. **Introduction**: Defines objectives and dataset details.
2. **Setup & Import**: Imports libraries (pandas, numpy, matplotlib, seaborn, sklearn).
3. **EDA**: Visualizes data distributions and correlations.
4. **Feature Engineering**: Imputes missing values, encodes categorical variables.
5. **Regression**: Applies Linear Regression (House Prices) and Logistic Regression (Titanic).
6. **Classification**: Implements SVM and KNN for Titanic survival prediction.
7. **Results**: Visualizes regression plots and confusion matrices.
8. **Conclusion**: Summarizes findings and suggests future improvements (e.g., hyperparameter tuning).

## Results
- **House Prices**: Linear Regression predicts `SalePrice` with an MSE (value printed in notebook).
- **Titanic**: Logistic Regression, SVM, and KNN predict `Survived` with accuracies (values printed in notebook).
- Visuals: Generated plots include correlation heatmaps, pairplots, regression scatter plots, and confusion matrices.

## Future Improvements
- Perform hyperparameter tuning for SVM and KNN.
- Include additional features or use feature selection techniques.
- Explore ensemble methods (e.g., Random Forest, Gradient Boosting).

## Notes
- Ensure `train.csv` for House Prices is in the working directory before running the notebook.
- The Titanic dataset is loaded automatically from the provided URL.
- For issues with PDF export, verify pandoc and LaTeX installations.