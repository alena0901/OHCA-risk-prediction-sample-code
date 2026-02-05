library(caret)
library(xgboost)
library(pROC)
library(dplyr)

set.seed(2025)
sampled_data <- readRDS("ohca_sampled_model_data_1500.RData")

data <- sampled_data

data$ohca2 <- as.integer(as.character(data$ohca2))
feature_cols <- setdiff(names(data), c("date", "ohca2"))

train_idx <- createDataPartition(data$ohca2, p = 0.8, list = FALSE)
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply model.matrix() separately to training and testing sets
# Note: If certain factor levels are not present in the test set but exist in the training set,
# the resulting column names may differ.
# This can cause XGBoost to throw an error during predict().
# In the original full dataset, this issue typically does not occur.

x_train <- as.matrix(train_data[, feature_cols])
x_test  <- as.matrix(test_data[, feature_cols])

x_train <- model.matrix(ohca2 ~ ., data = train_data)[, -1]
x_test <- model.matrix(ohca2 ~ ., data = test_data)[, -1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Recommended approach: combine training and testing sets before applying model.matrix()
# This ensures that the column names are consistent across both sets.

all_data <- rbind(train_data, test_data)
x_all <- model.matrix(ohca2 ~ ., all_data)[, -1]

x_train <- x_all[1:nrow(train_data), ]
x_test  <- x_all[(nrow(train_data)+1):nrow(all_data), ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dtrain <- xgb.DMatrix(x_train, label = train_data$ohca2)
dtest  <- xgb.DMatrix(x_test, label = test_data$ohca2)


param_grid <- expand.grid(
  eta = c(0.05, 0.1),
  max_depth = c(3, 6),
  min_child_weight = c(1, 5),
  subsample = c(0.8),
  colsample_bytree = c(0.8),
  gamma = c(0, 0.1)
)

grid_results <- lapply(1:nrow(param_grid), function(i) {
  
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    min_child_weight = param_grid$min_child_weight[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    gamma = param_grid$gamma[i]
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 500,
    nfold = 5,                
    stratified = TRUE,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  data.frame(
    param_grid[i, ],
    best_auc = max(cv$evaluation_log$test_auc_mean),
    best_nrounds = cv$best_iteration
  )
})

grid_results <- bind_rows(grid_results)


best_row <- grid_results[which.max(grid_results$best_auc), ]

best_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = best_row$eta,
  max_depth = best_row$max_depth,
  min_child_weight = best_row$min_child_weight,
  subsample = best_row$subsample,
  colsample_bytree = best_row$colsample_bytree,
  gamma = best_row$gamma
)

final_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_row$best_nrounds,
  verbose = 0
)


test_prob <- predict(final_model, dtest)

roc_test <- roc(test_data$ohca2, test_prob)
auc_test <- auc(roc_test)


cutoff <- coords(roc_test, "best", best.method = "youden")$threshold

spec_levels <- c(0.99, 0.95, 0.90)

fixed_spec_results <- lapply(spec_levels, function(s) {
  coords(roc_test, x = s, input = "specificity",
         ret = c("sensitivity", "specificity"))
})

fixed_spec_results

