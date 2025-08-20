# =============================================
# Machine Learning for Risk Prediction
# Author: Jeet Patel
# Date: 2025-08-25
# =============================================


# 0) Clear workspace and load packages
rm(list = ls())

install.packages(c("haven", "tidyverse", "tidymodels", "vip"))

library(haven)        # for reading SAS files
library(tidyverse)    # data manipulation & visualization
library(tidymodels)   # modeling & machine learning
library(vip)          # variable importance plots

# 1) Set working directory and load data
setwd("/Users/jeetpatel/Desktop/Final Project")

adsl <- read_sas("adsl.sas7bdat")
load("adae.Rda")  

# 2) Merge datasets
data <- left_join(adsl, adae, by = "USUBJID")

##Data Preparation##

# 3) Select relevant columns and rename
data <- data[, c("USUBJID", "AGE.y", "SEX.y", "RACE.y", "DCDECOD")]
names(data)[names(data) == "AGE.y"] <- "AGE"
names(data)[names(data) == "SEX.y"] <- "SEX"
names(data)[names(data) == "RACE.y"] <- "RACE"

# 4) Handle missing values
data$AGE[is.na(data$AGE)] <- median(data$AGE, na.rm = TRUE)
data$SEX  <- fct_na_value_to_level(data$SEX, "Unknown")
data$RACE <- fct_na_value_to_level(data$RACE, "Unknown")

# 5) Cap AGE outliers at 1st and 99th percentile
age_lower <- quantile(data$AGE, 0.01)
age_upper <- quantile(data$AGE, 0.99)
data$AGE[data$AGE < age_lower] <- age_lower
data$AGE[data$AGE > age_upper] <- age_upper

# 6) Remove rows with missing DCDECOD
data <- data[!is.na(data$DCDECOD), ]

# 7) Create binary target variable and remove original column
data$completed_flag <- factor(ifelse(data$DCDECOD == "COMPLETED", "Yes", "No"))
data <- data[, !(names(data) %in% "DCDECOD")]

##Exploratory Data Analysis##

# 8) Explore the dataset
str(data)
glimpse(data)
summary(data)
table(data$completed_flag)
prop.table(table(data$completed_flag))

# Barplot for completed_flag
completed_counts <- table(data$completed_flag)
barplot(completed_counts,
        horiz = TRUE,
        col = c("steelblue", "orange"),
        main = "Distribution of Completed Flag",
        xlab = "Count")


##Machine Learning Model##

# 9) Split data into training and testing sets
set.seed(123)  # reproducibility
split <- initial_split(data, prop = 0.8, strata = completed_flag)
train <- training(split)
test  <- testing(split)

# Confirm split
prop.table(table(train$completed_flag))

# 10) Create recipe for preprocessing
rec <- recipe(completed_flag ~ AGE + SEX + RACE, data = train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 11) Logistic Regression
log_model <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

log_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(log_model)

log_fit <- fit(log_wf, data = train)

# 12) Random Forest
rf_model <- rand_forest(mode = "classification", trees = 500) %>%
  set_engine("ranger", importance = "impurity")

rf_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_model)

rf_fit <- fit(rf_wf, data = train)

# 13) Evaluate Logistic Regression
log_preds <- bind_cols(predict(log_fit, test), select(test, completed_flag))
log_metrics <- metrics(log_preds, truth = completed_flag, estimate = .pred_class)
log_cm      <- conf_mat(log_preds, truth = completed_flag, estimate = .pred_class)

# 14) Evaluate Random Forest
rf_preds <- bind_cols(predict(rf_fit, test), select(test, completed_flag))
rf_metrics <- metrics(rf_preds, truth = completed_flag, estimate = .pred_class)
rf_cm      <- conf_mat(rf_preds, truth = completed_flag, estimate = .pred_class)

# 15) Feature importance for Random Forest
rf_fit_parsnip <- extract_fit_parsnip(rf_fit)
vip(rf_fit_parsnip)
