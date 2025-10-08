setwd("I:\\alena\\5 OHCA_climate_modle")


load("sampled_data.RData")

model_data <- sampled_data

model_data_transformed <- copy(model_data)  


char_cols <- names(model_data)[sapply(model_data, is.character)]  
factor_cols <- names(model_data)[sapply(model_data, is.factor)]  


for (col in char_cols) {  
  model_data_transformed[[col]] <- as.factor(model_data_transformed[[col]])  
}  


for (col in c(char_cols, factor_cols)) {  
  model_data_transformed[[col]] <- as.numeric(model_data_transformed[[col]])  
}  

 
for (col in names(model_data_transformed)) {  
  if (!is.numeric(model_data_transformed[[col]])) {  
    model_data_transformed[[col]] <- as.numeric(model_data_transformed[[col]])  
  }  
}  


set.seed(2025)  
trainIndex <- createDataPartition(model_data_transformed$ohca2, p = 0.8, list = FALSE)  
train_data <- model_data_transformed[trainIndex, ]  
test_data <- model_data_transformed[-trainIndex, ]  
table(test_data$ohca2)
 
feature_cols <- setdiff(names(train_data), c("date", "ohca2"))  #"target_variable",

 
train_matrix <- as.matrix(sapply(train_data[, ..feature_cols], as.numeric))  
test_matrix <- as.matrix(sapply(test_data[, ..feature_cols], as.numeric))  


dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_data$ohca2))  
dtest <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_data$ohca2))  


library(xgboost)  
library(dplyr)  
library(parallel)  
library(doParallel)  
library(foreach)  

 
results <- data.frame(  
  eta = numeric(),  
  max_depth = numeric(),  
  min_child_weight = numeric(),  
  subsample = numeric(),  
  colsample_bytree = numeric(),  
  best_auc = numeric(),  
  best_round = numeric()  
)  


library(xgboost)  
library(doParallel)  
library(foreach)  


num_cores <- min(detectCores() - 1, 4)   
cl <- makeCluster(num_cores)  
registerDoParallel(cl)  

  
 
param_grid <- expand.grid(  
  eta = c(0.01, 0.05, 0.1, 0.3),  
  max_depth = c(3, 6, 9),  
  min_child_weight = c(1, 3, 5),  
  subsample = c(0.6, 0.8, 1.0),  
  colsample_bytree = c(0.6, 0.8, 1.0),  
  gamma = c(0, 0.1, 0.3)  
)  




batch_size <- 20 
all_results <- list()  

  
start_time <- Sys.time()  

for (batch_start in seq(1, nrow(param_grid), by = batch_size)) {  


  clusterExport(cl, c("x_train_sample", "y_train_sample"), envir = environment())  
  

  batch_results <- foreach(  
    i = batch_start:batch_end,  
    .combine = rbind,  
    .packages = c("xgboost")  
  ) %dopar% {  
    dtrain_local <- xgb.DMatrix(data = x_train_sample, label = y_train_sample)  
    
    current_params <- list(  
      objective = "binary:logistic",  
      eval_metric = "auc",  
      eta = param_grid$eta[i],  
      max_depth = param_grid$max_depth[i],  
      min_child_weight = param_grid$min_child_weight[i],  
      subsample = param_grid$subsample[i],  
      colsample_bytree = param_grid$colsample_bytree[i],  
      gamma = param_grid$gamma[i]  
    )  
    

    cv_results <- xgb.cv(  
      params = current_params,  
      data = dtrain_local,  
      nrounds = 500, 
      nfold = 3,    
      stratified = TRUE,  
      early_stopping_rounds = 10,  
      verbose = FALSE,  
      metrics = "auc"  
    )  
    

    best_auc <- max(cv_results$evaluation_log$test_auc_mean)  
    best_round <- which.max(cv_results$evaluation_log$test_auc_mean)  
    
    data.frame(  
      eta = current_params$eta,  
      max_depth = current_params$max_depth,  
      min_child_weight = current_params$min_child_weight,  
      subsample = current_params$subsample,  
      colsample_bytree = current_params$colsample_bytree,  
      gamma = current_params$gamma,  
      best_auc = best_auc,  
      best_round = best_round  
    )  
  }  
  

  all_results[[length(all_results) + 1]] <- batch_results  
  
  
  saveRDS(do.call(rbind, all_results), "grid_search_results_checkpoint.rds")  
  

  elapsed <- difftime(Sys.time(), start_time, units = "mins")  
  progress <- batch_end / nrow(param_grid)  
  estimated_total <- elapsed / progress  
  remaining <- estimated_total - elapsed  

}  

grid_search_results <- do.call(rbind, all_results)  


stopCluster(cl)  

 
best_params <- grid_search_results[which.max(grid_search_results$best_auc), ]  

 
sorted_results <- grid_search_results[order(-grid_search_results$best_auc), ]  

 
print(best_params)  

print(difftime(Sys.time(), start_time, units = "mins"))  


saveRDS(grid_search_results, "final_grid_search_results.rds")  
saveRDS(best_params, "best_xgboost_params.rds")  


print(head(sorted_results, 10))  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

grid_search_results <- readRDS("final_grid_search_results.rds")
grid_search_results_sorted <- grid_search_results %>%  
  arrange(desc(best_auc))  



best_params <- list(  
  objective = "binary:logistic",  
  eval_metric = "auc",  
  eta = grid_search_results_sorted$eta[1],  
  max_depth = grid_search_results_sorted$max_depth[1],  
  min_child_weight = grid_search_results_sorted$min_child_weight[1],  
  subsample = grid_search_results_sorted$subsample[1],  
  colsample_bytree = grid_search_results_sorted$colsample_bytree[1],  
  gamma = grid_search_results_sorted$gamma[1]  
)  


final_model <- xgboost(  
  data = dtrain,  
  params = best_params,  
  nrounds = grid_search_results_sorted$best_round[1],  
  verbose = 1  
)  


stopCluster(cl)  


grid_search_results_sorted <- grid_search_results %>%  
  arrange(desc(best_auc))  


print("Top 5 Best Parameter Combinations:")  
print(head(grid_search_results_sorted, 5))  


best_params <- list(  
  objective = "binary:logistic",  
  eval_metric = "auc",  
  eta = grid_search_results_sorted$eta[1],  
  max_depth = grid_search_results_sorted$max_depth[1],  
  min_child_weight = grid_search_results_sorted$min_child_weight[1],  
  subsample = grid_search_results_sorted$subsample[1],  
  colsample_bytree = grid_search_results_sorted$colsample_bytree[1]  
)  


xgb_model_final <- xgboost(  
  data = dtrain,  
  params = best_params,  
  nrounds = grid_search_results_sorted$best_round[1],  
  verbose = 1  
)  



preds_val_prob <- predict(xgb_model_final, dtrain)


roc_result_val <- pROC::roc(train_data$ohca2, preds_val_prob)
auc_val <- pROC::auc(roc_result_val)

print(paste("test_AUC:", auc_val))


# xgb_cutoff <- coords(roc_result_val, "best", best.method = "youden")  
# xgb_cutoff

# ~~~~~~~~~~~~~~
preds_test_prob <- predict(xgb_model_final, dtest)  


preds_class <- ifelse(preds_test_prob > xgb_cutoff, 1, 0)  
true_class <- test_data$ohca2  


roc_result <- roc(true_class, preds_test_prob)  


conf_matrix <- confusionMatrix(factor(preds_class), factor(true_class))  


performance_metrics <- list(  
  AUC = auc(roc_result),  
  Sensitivity = conf_matrix$byClass["Sensitivity"],  
  Specificity = conf_matrix$byClass["Specificity"],  
  Precision = conf_matrix$byClass["Precision"],  
  Recall = conf_matrix$byClass["Recall"],  
  F1_Score = conf_matrix$byClass["F1"]  
)  

print(performance_metrics)  


roc_plot_data <- data.frame(  
  x = 1 - roc_result$specificities,  
  y = roc_result$sensitivities  
)  

roc_curve <- ggplot(roc_plot_data, aes(x = x, y = y)) +  
  geom_line(color = "blue", size = 1) +  
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +  
  labs(  
    title = paste("ROC Curve (AUC =", round(performance_metrics$AUC, 3), ")"),  
    x = "1 - Specificity (False Positive Rate)",  
    y = "Sensitivity (True Positive Rate)"  
  ) +  
  theme_minimal() +  
  coord_fixed()  


print(roc_curve)  


feature_importance <- xgb.importance(feature_names = feature_cols, model = xgb_model_final)  


importance_plot <- ggplot(feature_importance[1:min(50, nrow(feature_importance))],   
                          aes(x = reorder(Feature, Gain), y = Gain)) +  
  geom_bar(stat = "identity", fill = "steelblue") +  
  coord_flip() +  
  labs(  
    title = "Top Feature Importance",  
    x = "Features",  
    y = "Importance (Gain)"  
  ) +  
  theme_minimal()  

print(importance_plot)  

ggsave("importance_plot0326.png",importance_plot,width = 8,height = 11,dpi = 300)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
calculate_learning_curve <- function(model_data_transformed, feature_cols, train_sizes = seq(0.2, 0.8, by = 0.2)) {  

  learning_curve_results <- data.frame(  
    train_size = numeric(),  
    train_auc = numeric(),  
    test_auc = numeric()  
  )  
  

  for (size in train_sizes) {  
    set.seed(2025)  
    
 
    train_subset_index <- createDataPartition(model_data_transformed$ohca2, p = size, list = FALSE)  
    train_subset <- model_data_transformed[train_subset_index, ]  
    

    test_subset <- model_data_transformed[-train_subset_index, ]  
    
    train_matrix <- as.matrix(sapply(train_subset[, ..feature_cols], as.numeric))  
    test_matrix <- as.matrix(sapply(test_subset[, ..feature_cols], as.numeric))  

    dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_subset$ohca2))  
    dtest <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_subset$ohca2))  

    xgb_model <- xgb.train(  
      params = params,  
      data = dtrain,  
      nrounds = 100,  
      watchlist = list(train = dtrain, test = dtest),  
      early_stopping_rounds = 10,  
      verbose = 0  
    )  
    

    train_preds <- predict(xgb_model, dtrain)  
    test_preds <- predict(xgb_model, dtest)  
    

    train_auc <- auc(roc(train_subset$ohca2, train_preds))  
    test_auc <- auc(roc(test_subset$ohca2, test_preds))  
    

    learning_curve_results <- rbind(learning_curve_results,   
                                    data.frame(  
                                      train_size = size,  
                                      train_auc = train_auc,  
                                      test_auc = test_auc  
                                    ))  
  }  
  
  return(learning_curve_results)  
}  

# 计算学习曲线  
learning_curve_data <- calculate_learning_curve(model_data_transformed, feature_cols)  

# 绘制学习曲线  
learning_curve_plot <- ggplot(learning_curve_data) +  
  geom_line(aes(x = train_size, y = train_auc, color = "Training AUC"), size = 1) +  
  geom_point(aes(x = train_size, y = train_auc, color = "Training AUC"), size = 3) +  
  geom_line(aes(x = train_size, y = test_auc, color = "Testing AUC"), size = 1) +  
  geom_point(aes(x = train_size, y = test_auc, color = "Testing AUC"), size = 3) +  
  labs(  
    title = "Learning Curve for XGBoost Model",  
    x = "Proportion of Training Data",  
    y = "AUC Score",  
    color = "Data Set"  
  ) +  
  scale_color_manual(  
    values = c("Training AUC" = "blue", "Testing AUC" = "red")  
  ) +  
  theme_minimal() +  
  theme(legend.position = "bottom")  

 
print(learning_curve_plot)  
ggsave("learning_curve_xgb.png",learning_curve_plot,width = 8,height = 5,dpi = 300)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cv_log <- cv_results$evaluation_log


learning_curve_data <- data.frame(
  Iteration = 1:nrow(cv_log),
  Training_AUC = cv_log$train_auc_mean,
  Validation_AUC = cv_log$test_auc_mean,
  Training_AUC_SD = cv_log$train_auc_std,
  Validation_AUC_SD = cv_log$test_auc_std
)


learning_curve_plot <- ggplot(learning_curve_data, aes(x = Iteration)) +
  geom_line(aes(y = Training_AUC, color = "Training"), size = 1) +
  geom_ribbon(aes(ymin = Training_AUC - Training_AUC_SD, 
                  ymax = Training_AUC + Training_AUC_SD, 
                  fill = "Training"), alpha = 0.1) +
  geom_line(aes(y = Validation_AUC, color = "Validation"), size = 1) +
  geom_ribbon(aes(ymin = Validation_AUC - Validation_AUC_SD, 
                  ymax = Validation_AUC + Validation_AUC_SD, 
                  fill = "Validation"), alpha = 0.1) +
  geom_vline(xintercept = best_nrounds, linetype = "dashed", color = "darkred") +
  scale_color_manual(values = c("Training" = "blue", "Validation" = "red")) +
  scale_fill_manual(values = c("Training" = "blue", "Validation" = "red")) +
  labs(
    title = "Learning Curve - AUC",
    subtitle = paste("Best iteration at", best_nrounds),
    x = "Boosting Iteration",
    y = "AUC",
    color = "Dataset",
    fill = "Dataset"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(learning_curve_plot)


ggsave("learning_curve_plot_xgb.png",learning_curve_plot, width = 12, height = 8,dpi = 300)



if ("train_logloss_mean" %in% colnames(cv_log)) {

  loss_curve_data <- data.frame(
    Iteration = 1:nrow(cv_log),
    Training_Loss = cv_log$train_logloss_mean,
    Validation_Loss = cv_log$test_logloss_mean,
    Training_Loss_SD = cv_log$train_logloss_std,
    Validation_Loss_SD = cv_log$test_logloss_std
  )
  

  loss_curve_plot <- ggplot(loss_curve_data_2, aes(x = Iteration)) +
    geom_line(aes(y = Training_Loss, color = "Training"), size = 1) +
    geom_ribbon(aes(ymin = Training_Loss - Training_Loss_SD, 
                    ymax = Training_Loss + Training_Loss_SD, 
                    fill = "Training"), alpha = 0.1) +
    geom_line(aes(y = Validation_Loss, color = "Validation"), size = 1) +
    geom_ribbon(aes(ymin = Validation_Loss - Validation_Loss_SD, 
                    ymax = Validation_Loss + Validation_Loss_SD, 
                    fill = "Validation"), alpha = 0.1) +
    geom_vline(xintercept = best_nrounds, linetype = "dashed", color = "darkred") +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red")) +
    scale_fill_manual(values = c("Training" = "blue", "Validation" = "red")) +
    labs(
      title = "Loss Function Curve",
      subtitle = paste("Best iteration at", best_nrounds),
      x = "Boosting Iteration",
      y = "Log Loss",
      color = "Dataset",
      fill = "Dataset"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(loss_curve_plot)
} else {

  

  cv_params_with_logloss <- cv_params
  cv_params_with_logloss$eval_metric <- c("auc", "logloss")
  
  cv_results_with_logloss <- xgb.cv(
    params = cv_params_with_logloss,
    data = dtrain,
    nrounds = best_nrounds + 20,    
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 10,
    verbose = 0,
    metrics = c("auc", "logloss")
  )
  

  loss_log <- cv_results_with_logloss$evaluation_log
  

  loss_curve_data <- data.frame(
    Iteration = 1:nrow(loss_log),
    Training_Loss = loss_log$train_logloss_mean,
    Validation_Loss = loss_log$test_logloss_mean,
    Training_Loss_SD = loss_log$train_logloss_std,
    Validation_Loss_SD = loss_log$test_logloss_std
  )
  

  loss_curve_plot <- ggplot(loss_curve_data, aes(x = Iteration)) +
    geom_line(aes(y = Training_Loss, color = "Training"), size = 1) +
    geom_ribbon(aes(ymin = Training_Loss - Training_Loss_SD, 
                    ymax = Training_Loss + Training_Loss_SD, 
                    fill = "Training"), alpha = 0.1) +
    geom_line(aes(y = Validation_Loss, color = "Validation"), size = 1) +
    geom_ribbon(aes(ymin = Validation_Loss - Validation_Loss_SD, 
                    ymax = Validation_Loss + Validation_Loss_SD, 
                    fill = "Validation"), alpha = 0.1) +
    geom_vline(xintercept = best_nrounds, linetype = "dashed", color = "darkred") +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red")) +
    scale_fill_manual(values = c("Training" = "blue", "Validation" = "red")) +
    labs(
      title = "Loss Function Curve",
      subtitle = paste("Best iteration at", best_nrounds),
      x = "Boosting Iteration",
      y = "Log Loss",
      color = "Dataset",
      fill = "Dataset"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  print(loss_curve_plot)
}

ggsave("loss_curve_plot_xgb.png",loss_curve_plot, width = 12, height = 8,dpi = 300)



train_preds <- predict(xgb_model_final, dtrain)
test_preds <- predict(xgb_model_final, dtest)


train_residuals <- as.numeric(train_data$ohca2) - train_preds
test_residuals <- as.numeric(test_data$ohca2) - test_preds


residual_data <- data.frame(
  Predicted = c(train_preds, test_preds),
  Residual = c(train_residuals, test_residuals),
  Dataset = c(rep("Training", length(train_preds)), rep("Testing", length(test_preds)))
)

residual_plot <- ggplot(residual_data, aes(x = Predicted, y = Residual, color = Dataset)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "darkred") +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = c("Training" = "blue", "Testing" = "red")) +
  labs(
    title = "Residual Plot",
    x = "Predicted Probability",
    y = "Residual (Actual - Predicted)",
    color = "Dataset"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")


print(residual_plot)

ggsave("residual_plot_xgb.png",residual_plot, width = 12, height = 8,dpi = 300)



performance_comparison <- data.frame(
  Metric = c("AUC", "Log Loss"),
  Training = c(
    max(cv_log$train_auc_mean),
    if("train_logloss_mean" %in% colnames(cv_log)) min(cv_log$train_logloss_mean) else NA
  ),
  Validation = c(
    max(cv_log$test_auc_mean),
    if("test_logloss_mean" %in% colnames(cv_log)) min(cv_log$test_logloss_mean) else NA
  ),
  Difference = c(
    max(cv_log$train_auc_mean) - max(cv_log$test_auc_mean),
    if("train_logloss_mean" %in% colnames(cv_log)) 
      min(cv_log$test_logloss_mean) - min(cv_log$train_logloss_mean) 
    else NA
  )
)



print(performance_comparison)

auc_diff <- performance_comparison$Difference[1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 保存结果  
saveRDS(list(  
  model = xgb_model_final, 
  finalmodel = xgb_model_final,
  performance_metrics = performance_metrics,  
  roc_result = roc_result,  
  feature_importance = feature_importance,
  # cutoff = xgb_cutoff,
  # learning_curve_data = learning_curve_data,
  # residual_data=residual_data,
  # loss_curve_data=loss_curve_data,
  # learning_curve_data_1=learning_curve_data_1,
  params = best_params  
), paste0("xgb_model_results_", k, "days.rds"))  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


library(xgboost)  
library(SHAPforxgboost)  
 
shap_values <- shap.values(  
  xgb_model = xgb_model,  
  X_train = dtrain 
)  


shap_matrix <- shap_values$shap_score 
print(dim(shap_matrix)) 



shap_long <- shap.prep(  
  shap_contrib = shap_values$shap_score,   
  X_train = train_data[, ..feature_cols]  #train_features 
)  



library(doParallel)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)


p <- shap.plot.summary(shap_long)  

ggsave("sv_dependence_p_0325.png",p, width = 12, height = 15,dpi = 300)

stopCluster(cl)



