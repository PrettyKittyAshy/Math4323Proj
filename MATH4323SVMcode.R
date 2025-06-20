# Load packages
library(readr)
library(e1071)
library(caret)

# Load and clean data
data <- read_csv("C:/Users/prett/Downloads/Student Depression Dataset.csv")
head(data)

student_depression_cleaned <- subset(data, select = -c(`Work Pressure`, `Job Satisfaction`))
student_depression_cleaned <- student_depression_cleaned[student_depression_cleaned$Profession == "Student", ]
student_depression_cleaned <- subset(student_depression_cleaned, select = -c(Profession, id))
names(student_depression_cleaned)[10] <- "Suicidal.Thoughts" 

# Handle missing values
colSums(is.na(student_depression_cleaned))
student_depression_cleaned$`Financial Stress`[is.na(student_depression_cleaned$`Financial Stress`)] <- 3

# Scale numeric variables
student_depression_cleaned$Age <- scale(student_depression_cleaned$Age)
names(student_depression_cleaned)[2] <- "Age"

# Convert factors
factor_cols <- c("Gender", "City", "Sleep Duration", "Dietary Habits", 
                 "Degree", "Suicidal.Thoughts", "Family History of Mental Illness", "Depression")
student_depression_cleaned[factor_cols] <- lapply(student_depression_cleaned[factor_cols], as.factor)

# Check for duplicates
any(duplicated(student_depression_cleaned))

# NEW CODE: 80/20 TRAIN-TEST SPLIT

set.seed(123)
split_index <- createDataPartition(student_depression_cleaned$Depression, 
                                   p = 0.8, list = FALSE)
train_data <- student_depression_cleaned[split_index, ]
test_data <- student_depression_cleaned[-split_index, ]

# SVM MODEL WITH TUNING (WITH PROGRESS)

# Reduced tuning grid
tune_grid <- expand.grid(
  cost = c(0.1, 1, 10),
  gamma = c(0.01, 0.1)
)

# Manual progress-tracking version
set.seed(123)
results <- list()
combinations <- nrow(tune_grid)

cat(paste("Tuning progress (0/", combinations, ")\n", sep=""))

for(i in 1:nrow(tune_grid)) {
  # Print progress
  cat(paste("Testing combination ", i, "/", combinations, 
            " (cost=", tune_grid$cost[i], 
            ", gamma=", tune_grid$gamma[i], ")\n", sep=""))
  
  # Train model
  model <- svm(
    Depression ~ .,
    data = train_data,
    kernel = "radial",
    cost = tune_grid$cost[i],
    gamma = tune_grid$gamma[i],
    cross = 3  # 3-fold CV
  )
  
  # Store results
  results[[i]] <- list(
    cost = tune_grid$cost[i],
    gamma = tune_grid$gamma[i],
    accuracy = 1 - model$tot.accuracy/100  # Convert to error rate
  )
}

# Convert results to data frame
tune_results <- do.call(rbind, lapply(results, as.data.frame))

# Find best parameters
best_params <- tune_results[which.min(tune_results$accuracy), ]

# Print the optimal values:
cat("\nOptimal SVM Parameters:\n")
cat("Cost (C):", best_params$cost, "\n")
cat("Gamma (γ):", best_params$gamma, "\n")

# Train final model with best parameters
best_svm <- svm(
  Depression ~ .,
  data = train_data,
  kernel = "radial",
  cost = best_params$cost,
  gamma = best_params$gamma,
  probability = TRUE
)

# Print tuning results
print(best_svm)

# Evaluate on test set
svm_pred <- predict(best_svm, test_data)
conf_matrix <- confusionMatrix(svm_pred, test_data$Depression)

# Print results
print(conf_matrix)
cat("Tuned SVM Accuracy:", conf_matrix$overall['Accuracy'], "\n")

# Compare with default SVM
default_svm <- svm(Depression ~ ., data = train_data, kernel = "radial")
default_pred <- predict(default_svm, test_data)
default_acc <- mean(default_pred == test_data$Depression)
cat("Default SVM Accuracy:", default_acc, "\n")

# FINAL SVM MODEL ON FULL DATASET WITH OPTIMAL PARAMETERS

# Train final model on full dataset with best parameters
final_svm <- svm(
  Depression ~ .,
  data = student_depression_cleaned,  # Using full dataset
  kernel = "radial",
  cost = 1,        # Your optimal cost
  gamma = 0.01,    # Your optimal gamma
  probability = TRUE
)

# 1. Summary of SVM object
cat("\nSUMMARY OF FINAL SVM MODEL:\n")
print(summary(final_svm))

# 2. Support vectors in each class
cat("\nNUMBER OF SUPPORT VECTORS IN EACH CLASS:\n")
print(final_svm$nSV)

library(caret)

# Set up 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Re-train with cross-validation
svm_cv <- train(
  Depression ~ .,
  data = student_depression_cleaned,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = data.frame(C = 1, sigma = 0.01),  # Your optimal parameters
  metric = "Accuracy"
)

# Print CV results
cat("\nCROSS-VALIDATION RESULTS:\n")
print(svm_cv$results)

# Key metrics:
cat("\nMEAN CV ACCURACY:", mean(svm_cv$resample$Accuracy), "\n")
cat("SD OF ACCURACY:", sd(svm_cv$resample$Accuracy), "\n")

# Predictions on same data (tends to be overly optimistic)
preds <- predict(final_svm, student_depression_cleaned)

# Confusion matrix
conf_mat <- confusionMatrix(preds, student_depression_cleaned$Depression)

cat("\nCONFUSION MATRIX (FULL DATA):\n")
print(conf_mat)

# Key metrics:
cat("\nOVERALL ACCURACY:", conf_mat$overall['Accuracy'], "\n")
cat("CLASS-SPECIFIC METRICS:\n")
print(conf_mat$byClass)

# In order to see some of the boundaries that ended up drawn, we segment out two predictors
names(student_depression_cleaned) <- make.names(names(student_depression_cleaned))
predictors <- c("Study.Satisfaction", "Work.Study.Hours")
svm_data_subset <- student_depression_cleaned[, c("Depression", predictors)]

# Train SVM on two selected predictors, whatever boundary you want to see
svm_model_2d <- svm(
  Depression ~ .,
  data = svm_data_subset,
  kernel = "radial",
  cost = 1,
  gamma = 0.01,
  probability = TRUE
)

# Plot the decision boundary
plot(svm_model_2d, svm_data_subset, main = "Specific Decision Boundary")
# This proved fairly useless :( Talked about it in the report

# Create dummy variables (basically, split all factors into numericals)
dummy_data <- dummyVars(Depression ~ ., data = student_depression_cleaned)
X_dummy <- predict(dummy_data, newdata = student_depression_cleaned)
X_df <- as.data.frame(X_dummy)

# Put the response variable into this new data set
X_df$Depression <- student_depression_cleaned$Depression

# Train a linear SVM on the all-numeric values
linear_svm <- svm(
  Depression ~ ., 
  data = X_df, 
  kernel = "linear", 
  cost = 1,
  scale = FALSE
)

# get all the coefficients on the model
weights <- t(linear_svm$coefs) %*% linear_svm$SV
weights_df <- data.frame(
  Feature = colnames(X_df)[colnames(X_df) != "Depression"],
  Importance = as.vector(abs(weights))
)
weights_df <- weights_df[order(-weights_df$Importance), ]
head(weights_df, 10)