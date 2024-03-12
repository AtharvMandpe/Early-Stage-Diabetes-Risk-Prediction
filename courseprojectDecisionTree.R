getwd()
setwd("C:/Users/athar/Desktop/oversampling")

#install.packages("glmnet")
#install.packages("caret")
library(glmnet)
library(caret)
library(rpart)
library(randomForest)

data <- read.csv("oversampled_data_rose.csv")


set.seed(123)

outcome_variable <- data$class

train_indices <- createDataPartition(outcome_variable, p = 0.75, list = FALSE)

train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

train_data <- na.omit(train_data)
test_data <- na.omit(test_data)
test_data_without_outcome <- test_data[, -which(names(test_data) == "class")]

ctrl <- trainControl(method = "cv", number = 10)
formula <- class ~ .

#dt model----
model <- train(formula,
               data = train_data, 
               method = "rpart",
               trControl = ctrl)

print(model)


predictions <- predict(model, newdata = test_data_without_outcome)


table(predictions)
table(test_data$class)


predictions <- factor(predictions)
test_data$class <- factor(test_data$class)



conf_mat <- confusionMatrix(predictions, test_data$class)


print(conf_mat)



#random forest model----
model <- train(formula,
               data = train_data, 
               method = "rf",
               trControl = ctrl)

rf_model <- model$finalModel

print(model)


importance <- importance(rf_model)
print(importance)
varImpPlot(rf_model)



predictions <- predict(model, newdata = test_data_without_outcome)


table(predictions)
table(test_data$class)


predictions <- factor(predictions)
test_data$class <- factor(test_data$class)



conf_mat <- confusionMatrix(predictions, test_data$class)


print(conf_mat)



#lr model----
model <- train(formula,
               data = train_data, 
               method = "glm",
               trControl = ctrl,
               family = binomial)


print(model)


predictions <- predict(model, newdata = test_data_without_outcome, type="prob")
threshold <- 0.5
predictions <- ifelse(predictions[, "Positive"] > threshold, "Positive", "Negative")

test_data$class <- factor(test_data$class, levels = c("Negative", "Positive"))
predictions <- factor(predictions, levels = c("Negative", "Positive"))

print(typeof(test_data$class))
print(typeof(predictions))



conf_mat <- confusionMatrix(predictions, test_data$class)
print(conf_mat)

#svm----
model <- train(formula,
               data = train_data, 
               method = "svmLinear",
               trControl = ctrl)

print(model)

predictions <- predict(model, newdata = test_data_without_outcome)

conf_mat <- confusionMatrix(predictions, test_data$class)


print(conf_mat)


# Lasso regression model----
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

model <- train(
  formula,
  data = train_data, 
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, length = 100)), # Lasso regression (alpha = 1)
  preProc = c("center", "scale"),
  metric = "Accuracy",
  tuneLength = 10
)

print(model)

lasso_coef <- coef(model$finalModel)
lasso_coef
non_zero_indices <- which(lasso_coef != 0, arr.ind = TRUE)

# Extract selected features based on non-zero indices
selected_features <- rownames(lasso_coef)[non_zero_indices[,1]]
selected_features



predictions <- predict(model, newdata = test_data_without_outcome)

conf_mat <- confusionMatrix(predictions, test_data$class)

print(conf_mat)
