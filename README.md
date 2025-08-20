# Project Overview

This project involved developing a machine learning model to predict patient outcomes using real-world healthcare data. The objective was to gain hands-on experience with **data preprocessing, exploratory analysis, and supervised machine learning**, while producing interpretable insights with potential clinical relevance.

---

# Objectives

- Prepare and clean the healthcare dataset for analysis, addressing missing values, outliers, and inconsistencies.  
- Perform exploratory data analysis (EDA) to understand feature distributions and relationships.  
- Build and evaluate machine learning models for outcome prediction.  
- Document the entire workflow in a **reproducible and transparent format**.  

---

# Methodology

## Data Preparation

- Merged two datasets (`adsl` and `adae`) to create a comprehensive analysis dataset.  
- Selected relevant variables and standardized column names for clarity.  
- Addressed missing values by replacing missing numeric values with medians and categorical values with an “Unknown” category.  
- Capped numeric outliers at the 1st and 99th percentiles to reduce the influence of extreme values.  
- Removed records with missing outcome information.  

## Exploratory Data Analysis

- Conducted statistical summaries and visualizations to assess feature distributions.  
- Examined the distribution of the target variable (`completed_flag`) and identified class imbalances.  

## Machine Learning Modeling

- Defined a binary outcome variable for study completion.  
- Preprocessed predictors using **recipes**:  
  - Converted categorical features into dummy variables  
  - Standardized numeric features  
- Built and compared two supervised models:  
  - **Logistic Regression**: baseline model with interpretable coefficients  
  - **Random Forest**: flexible ensemble model capable of capturing complex patterns  
- Evaluated model performance using metrics and confusion matrices.  
- Generated feature importance plots for the Random Forest model.  

---

# Tools & Technologies

- **R Packages**: `tidyverse`, `tidymodels`, `haven`, `vip`  
- **Workflow Documentation**: RMarkdown for reproducibility and clarity  

---

# Key Learnings

- Practical experience in **cleaning and preparing real-world healthcare datasets**.  
- Hands-on understanding of **machine learning model development, evaluation, and interpretation**.  
- The importance of **reproducible workflows** in applied data science projects.  
- How to generate actionable insights that are interpretable in a clinical context.  

---

# Future Work

- Extend the analysis to **larger datasets** or multi-site patient data.  
- Explore additional ML algorithms (e.g., XGBoost, SVM) to improve predictive performance.  
- Incorporate more **clinical features** for richer modeling and interpretability.  
