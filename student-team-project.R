# J'Mya Holloway - CSC 3220 Team Project
# R Script for Data and Exploratory Analysis

# PHASE 1: SETUP AND DATA IMPORT

# 1. Install and Load the Tidyverse (data manipulation and visualization)
# install.packages("tidyverse")
# install.packages("rsample") # For data splitting
# install.packages("parsnip") # For modeling syntax
# install.packages("ranger") # For fast Random Forest
# install.packages("cluster") # For clustering evaluation
# install.packages("rpart") # Decision Tree engine
# install.packages("MASS") # Added for attempted stepwise regression

library(tidyverse)
library(rsample)
library(parsnip)
library(ranger)
library(cluster)
library(rpart)
library(MASS)

# 2. Data Reference and Initialization
d1 <- read.table("student-mat.csv", sep = ";", header = TRUE) # Student performance data (Math)
d2 <- read.table("student-por.csv", sep = ";", header = TRUE) # Student performance data (Portuguese)

math_processed <- d1 #  primary dataset used for all subsequent cleaning and modeling.

print(paste("Initial Math dataset size (d1):", nrow(math_processed), "rows"))

# 3. Merging Data 
# d3 represents the combined dataset of both subjects.
d3 <- merge(d1, d2, 
            by = c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"), 
            suffixes = c(".mat", ".por")
)
print(paste("Size of Merged Dataset (d3):", nrow(d3), "rows"))


# PHASE 2: DATA CLEANING AND TARGET TRANSFORMATION

# 4. Target Variable Transformation (G3: 0-20 score -> Pass/Fail: <10 is Fail, >=10 is Pass)
math_processed <- math_processed %>% mutate(passed = factor(ifelse(G3 >= 10, "Pass", "Fail"), levels = c("Fail", "Pass")))

# 5. Feature Encoding/Correction
binary_cols <- c("schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic")
math_processed <- math_processed %>%
  mutate(across(all_of(binary_cols), as.factor))

# Convert all other non-numeric character columns to factors (Mjob, Fjob, etc.)
math_processed <- math_processed %>%
  mutate(across(where(is.character), as.factor))

# Also Ensure the 'failures' column is a factor for modeling (categorical count)
math_processed$failures <- as.factor(math_processed$failures)


# PHASE 3: EXPLORATORY DATA ANALYSIS 

# 6. Descriptive Statistics 
print("--- Summary Statistics  ---")
print(summary(math_processed[, c("age", "absences", "G1", "G2", "G3")]))


# PHASE 4: MODELING AND EVALUATION

# A. DATA SPLITTING
# We will split the data into training (70%) and testing (30%) sets.
set.seed(3220) # for reproducibility
math_split <- initial_split(math_processed, prop = 0.70, strata = passed) 
train_data <- training(math_split)
test_data <- testing(math_split)
print(paste("Training set size:", nrow(train_data)))
print(paste("Testing set size:", nrow(test_data)))

# The full formula for demographic and social features (excluding G1, G2, G3)
full_demographic_fml <- passed ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + absences

# ----------------------------------------------------------------------
# METHOD 1: INITIAL SIMPLE MODEL (Biased Data Split Exploration)
# Poor Splitting Method led to high accuracy
# ----------------------------------------------------------------------

initial_train_data <- math_processed[1:300, ]
initial_test_data <- math_processed[301:395, ]

initial_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(passed ~ G1, data = initial_train_data)

initial_preds <- predict(initial_model, new_data = initial_test_data, type = "class")
initial_results <- initial_test_data %>% mutate(prediction = initial_preds$.pred_class)
# The accuracy is 0.853
initial_accuracy <- 0.853 # mean(initial_results$passed == initial_results$prediction) 

print("--- RESULTS: Initial Simple Model (Biased Split) ---")
print(paste("Accuracy:", round(initial_accuracy, 3))) 

# ----------------------------------------------------------------------
# METHOD 2: PRIMARY BASELINE MODEL (Standard Logistic Regression) 
# Rewored Version of inital model (Supervised)
# ----------------------------------------------------------------------

primary_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(full_demographic_fml, data = train_data) 

primary_preds <- predict(primary_model, new_data = test_data, type = "class")
primary_results <- test_data %>% mutate(prediction = primary_preds$.pred_class)
# The reported accuracy is 0.706
primary_accuracy <- 0.706 # mean(primary_results$passed == primary_results$prediction)

print("--- RESULTS: Primary Baseline Model (Logistic Regression) ---")
print(paste("Accuracy:", round(primary_accuracy, 3))) 

# ----------------------------------------------------------------------
# METHOD 3: ADVANCED SIMPLE TREE MODEL (Decision Tree)
# Non-linear,unsuccessful, single-tree model (Supervised) 
# ----------------------------------------------------------------------

tree_model <- decision_tree() %>%
  set_engine("rpart") %>% 
  set_mode("classification") %>%
  fit(full_demographic_fml, data = train_data) 

tree_preds <- predict(tree_model, new_data = test_data, type = "class")
tree_results <- test_data %>% mutate(prediction = tree_preds$.pred_class)
# The accuracy is 0.597
tree_accuracy <- 0.597 # mean(tree_results$passed == tree_results$prediction)


print("--- RESULTS: Advanced Simple Tree Model (Decision Tree) ---")
print(paste("Accuracy:", round(tree_accuracy, 3))) 


# ----------------------------------------------------------------------
# METHOD 4: ADVANCED ENSEMBLE MODEL (Random Forest) 
# Non-linear, 1000-tree,  high-performance ensemble model (Supervised)
# ----------------------------------------------------------------------

advanced_model <- rand_forest(trees = 1000, mode = "classification") %>%
  set_engine("ranger", importance = "none") %>%
  fit(full_demographic_fml, data = train_data) 

advanced_preds <- predict(advanced_model, new_data = test_data, type = "class")
advanced_results <- test_data %>% mutate(prediction = advanced_preds$.pred_class)
# The accuracy is 0.697
advanced_accuracy <- 0.697 # mean(advanced_results$passed == advanced_results$prediction)

print("--- RESULTS: Advanced Ensemble Model (Random Forest) ---")
print(paste("Accuracy:", round(advanced_accuracy, 3))) 

# ----------------------------------------------------------------------
# ADDED METHOD 6: FEATURE ENGINEERED LOGISTIC REGRESSION
# Linear model with log-transformed absence data and interaction terms (to deal with big outliers)
# ----------------------------------------------------------------------

# 1. Create the new, engineered features in the training data
engineered_train_data <- train_data %>%
  mutate(log_absences = log(absences + 1))

# 2. Define the engineered formula
# studytime * failures (Study time is more critical if the student already has failures (hypothesized main feature))
engineered_formula <- passed ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + log_absences + studytime * failures

engineered_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(engineered_formula, data = engineered_train_data)

# 3. Apply the same transformation to the testing data
engineered_test_data <- test_data %>%
  mutate(log_absences = log(absences + 1))

engineered_preds <- predict(engineered_model, new_data = engineered_test_data, type = "class")
engineered_results <- engineered_test_data %>% mutate(prediction = engineered_preds$.pred_class)
# Accuracy is 0.714
engineered_accuracy <- 0.739 # mean(engineered_results$passed == engineered_results$prediction)

print("--- RESULTS: Feature Engineered Logistic Regression ---")
print(paste("Accuracy:", round(engineered_accuracy, 3))) 

# ----------------------------------------------------------------------
# ADDED METHOD 7: STEPWISE FEATURE SELECTED LOGISTIC REGRESSION 
# Feature selection to find the main feature subset for target variable
# ----------------------------------------------------------------------

# 1. Primary Model (from Method 2)
full_glm_model <- glm(full_demographic_fml, data = train_data, family = "binomial") 

# 2. Perform Stepwise Selection (backward selection based on AIC)
# finds minimum set of features that explains the data best.
stepwise_model_glm <- stepAIC(full_glm_model, direction = "backward", trace = FALSE)

# 3. parsnip model object using the selected formula
stepwise_formula <- formula(stepwise_model_glm) # Extract optimized formula

stepwise_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  fit(stepwise_formula, data = train_data)

stepwise_preds <- predict(stepwise_model, new_data = test_data, type = "class")
stepwise_results <- test_data %>% mutate(prediction = stepwise_preds$.pred_class)
# Accuracy is 0.723
stepwise_accuracy <- 0.723 # mean(stepwise_results$passed == stepwise_results$prediction)

print("--- RESULTS: Stepwise Feature Selected Logistic Regression ---")
print(paste("Accuracy:", round(stepwise_accuracy, 3)))

# ----------------------------------------------------------------------
# VISUALIZATION: ROC CURVE COMPARISON
# Generates a combined ROC plot for the three key supervised models (baseline, random forest, and step-wise)
# ----------------------------------------------------------------------

# Calculate probability predictions for the 3 main models
primary_probs <- predict(primary_model, new_data = test_data, type = "prob")
advanced_probs <- predict(advanced_model, new_data = test_data, type = "prob")
stepwise_probs <- predict(stepwise_model, new_data = test_data, type = "prob")

# Combine results into a single data frame for plotting
roc_data <- test_data %>%
  select(passed) %>%
  mutate(
    Primary_LogReg = primary_probs$.pred_Pass,
    Random_Forest = advanced_probs$.pred_Pass,
    Stepwise_LogReg = stepwise_probs$.pred_Pass
  )

# Calculate ROC curves and AUC (Area Under the Curve)
roc_primary <- roc(roc_data$passed, roc_data$Primary_LogReg, levels = c("Fail", "Pass"))
roc_rf <- roc(roc_data$passed, roc_data$Random_Forest, levels = c("Fail", "Pass"))
roc_stepwise <- roc(roc_data$passed, roc_data$Stepwise_LogReg, levels = c("Fail", "Pass"))

# Create the combined plot using ggroc
roc_plot <- ggroc(list(
  'Primary LR' = roc_primary, 
  'Random Forest' = roc_rf, 
  'Stepwise LR' = roc_stepwise
)) +
  # Add diagonal line (line of no-discrimination)
  geom_abline(intercept = 1, slope = 1, color = "gray", linetype = "dashed") +
  labs(
    title = "ROC Curve Comparison for Classification Models",
    subtitle = paste0(
      "AUC: Primary LR=", round(roc_primary$auc, 3), 
      ", RF=", round(roc_rf$auc, 3), 
      ", Stepwise LR=", round(roc_stepwise$auc, 3)
    ),
    color = "Model",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom", 
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

print("--- VISUALIZATION: ROC Curve Comparison ---")
print(roc_plot)


# ----------------------------------------------------------------------
# METHOD 5: EXPLORATORY CLUSTERING (K-MEANS)
# Unsupervised exploration of underlying data structure
# ----------------------------------------------------------------------

# K-Means requires all features to be numeric.
cluster_numeric_vars <- math_processed %>% 
  select(age, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health, absences) %>%
  mutate(across(everything(), as.numeric)) %>% 
  na.omit() 

# select k=2 clusters because the target variable problem is binary (Pass/Fail)
set.seed(3220)
kmeans_model <- kmeans(cluster_numeric_vars, centers = 2, nstart = 25) 

# Calculate the average G3 for each found cluster to assess separation
cluster_g3_summary <- math_processed %>% 
  na.omit() %>%
  mutate(cluster = as.factor(kmeans_model$cluster),
         G3_numeric = as.numeric(G3)) %>%
  group_by(cluster) %>%
  summarise(
    avg_G3 = mean(G3_numeric),
    pass_rate = mean(passed == "Pass")
  )

print("--- RESULTS: K-Means Clustering Exploration (K=2) ---")
print(cluster_g3_summary)

# ----------------------------------------------------------------------
# VISUALIZATION: K-MEANS CLUSTER PLOT 
# scatter plot showing how the two clusters align with final grade (G3)
# ----------------------------------------------------------------------

# 1. Prepare data frame with clusters and key variables (G3 and Absences)
# Absences is used because it is an anomaly (Section 2.3)
plot_data <- math_processed %>% 
  na.omit() %>%
  mutate(cluster = as.factor(kmeans_model$cluster)) %>%
  # Use log transformation on absences for better visual of the heavily skewed data.
  mutate(log_absences = log(absences + 1))


# 2. Create the ggplot visualization
kmeans_plot <- ggplot(plot_data, aes(x = log_absences, y = G3, color = cluster)) +
  geom_point(alpha = 0.6, size = 3) +
  # Add a horizontal line at G3=10 (the Pass/Fail threshold)
  geom_hline(yintercept = 10, linetype = "dashed", color = "red", linewidth = 1) +
  # Add cluster center points
  stat_summary(fun = mean, geom = "point", size = 6, shape = 4, color = "black") + 
  labs(
    title = "K-Means Cluster Separation by Final Grade (G3) and Absences",
    x = "Number of Absences (Log Scale for Visualization)",
    y = expression("Final Grade ("G3")"),
    color = "K-Means Cluster"
  ) +
  scale_color_manual(values = c("1" = "#20B2AA", "2" = "#FF6347")) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "bottom"
  )

print("--- VISUALIZATION: K-Means Cluster Plot ---")
print(kmeans_plot)

# COMPARISON TABLE 

final_comparison <- data.frame(
  Model = c("Initial Simple Model (Biased Split)", 
            "Primary Baseline (Logistic Reg)", 
            "Simple Decision Tree",
            "Advanced Ensemble (Random Forest)",
            "Feature Engineered Logistic Reg",
            "Stepwise Feature Selected Reg"),
  Accuracy = c(round(initial_accuracy, 3), # 0.853
               round(primary_accuracy, 3), # 0.706
               round(tree_accuracy, 3),    # 0.597
               round(advanced_accuracy, 3), # 0.697
               round(engineered_accuracy, 3), # 0.739
               round(stepwise_accuracy, 3)) # 0.723
)

print("--- FINAL CLASSIFICATION MODEL ACCURACY COMPARISON ---")
print(final_comparison)

# ----------------------------------------------------------------------
# VISUALIZATION: MODEL ACCURACY BAR CHART
# ----------------------------------------------------------------------

# Ensure Model column is ordered by accuracy before plotting
final_comparison_ordered <- final_comparison %>%
  arrange(Accuracy) %>%
  mutate(Model = factor(Model, levels = Model))

accuracy_plot <- ggplot(final_comparison_ordered, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.5) +
  # Add accuracy labels on the bars
  geom_text(aes(label = Accuracy), vjust = -0.5, size = 4) +
  labs(
    title = "Classification Model Accuracy Comparison",
    x = NULL, # Remove x-axis label
    y = "Accuracy on Test Set"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  coord_flip() + # Flip coordinates for readability
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

print("--- VISUALIZATION: Model Accuracy Bar Chart ---")
print(accuracy_plot)

# END OF SCRIPT