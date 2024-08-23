# Cross Validation

# LIBRARIES
library(VIM)
library(lattice)
library(mice)
library(caret)
#library(randomForest)
#library(rpart)
#library(e1071)
#library(caTools)
#library(class)
#library(neuralnet)
#library(xgboost)

# Read Data
r_data <- read.csv(file = "data/raw_final.csv", sep = ',')
r_data <- r_data[, !names(r_data) %in% c("Height", "Weight", "Diagnosis", "date_dx", "date_death", "date_lc", "date_lf", "Survival_Months")]
r_data[sapply(r_data, is.character)] <- lapply(r_data[sapply(r_data, is.character)], as.factor)
r_data[sapply(r_data, is.integer)] <- lapply(r_data[sapply(r_data, is.integer)], as.numeric)

for (i in 1:ncol(r_data)) {
    names(r_data)[i] <- paste0("V", i)
}

# Data Shuffling
set.seed(123)
data = r_data[sample(1:nrow(r_data)),]
rownames(data) <- 1:nrow(data)

# Data Splitting
set.seed(123)
presplit <- data
split <- createDataPartition(y = presplit$V26, p = 0.8, list = FALSE)
c_train <- presplit[split,]
c_val <- presplit[-split,]

#library(dplyr)
#data %>% group_by(V1) %>% summarise(Percentage=n()/nrow(.)*100)

#mean(data$V27, na.rm = TRUE)
#sd(data$V27, na.rm = TRUE)
#range(data$V27, na.rm = TRUE)

# Check data missingess
#aggr(data, col = mdc(1:2), numbers = TRUE, sortVars = TRUE, labels = names(data), cex.axis = 0.53, cex.numbers = 0.6, gap = 3, ylab = c("Proportion of missingness", "Missingness Pattern"))

# PRE-PROCESS (TRAIN)
# Imputation
set.seed(123)
init <- mice(c_train, maxit = 0)
meth <- init$method
predM <- init$predictorMatrix

for (i in colnames(c_train))
{
    if (sapply(c_train[i], is.factor))
    {
        predM[, c(i)] <- 0
    }
    else
    {
       meth[c(i)] <- "pmm"
    }
}

imp_data <- mice(c_train, m = 5, maxit = 50, method = meth, predictorMatrix = predM, seed = 123)
comp_data <- complete(imp_data)
#train <- subset(comp_data, V27 >= 0)
train <- comp_data

# PRE-PROCESS (VAL)
set.seed(123)
val <- c_val
val <- c_val[complete.cases(c_val),]
rownames(val) <- 1:nrow(val)

# SAVE CSV FOR PYTHON TRANSFER
#write.csv(train, file = "data/train.csv", row.names = FALSE)
#write.csv(val, file = "data/valid.csv", row.names = FALSE)

# CROSS VALIDATION
folds <- 10
cvFolds <- createFolds(factor(train$V26), k = folds, returnTrain = FALSE)

train_control <- trainControl(
    index = cvFolds,
    method = "cv",
    verboseIter = TRUE,
    number = 10)

# RANDOM FOREST
# Training
set.seed(123)
rf_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "rf",
    trControl = train_control
)

# Validation
rf_pred <- predict(rf_fit, newdata = val)
rf_cm <- confusionMatrix(table(val$V26, rf_pred), mode = "everything")
print(rf_cm)

# DECISION TREE
# Training
set.seed(123)
dt_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "rpart",
    trControl = train_control
)

# Validation
dt_pred <- predict(dt_fit, val)
dt_cm <- confusionMatrix(table(val$V26, dt_pred), mode = "everything")
print(dt_cm)

# SUPPORT VECTOR MACHINE (SVM)
# Training
set.seed(123)
svm_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "svmLinear",
    scale = FALSE,
    trControl = train_control
)

# Validation
svm_pred <- predict(svm_fit, val)
svm_cm <- confusionMatrix(table(val$V26, svm_pred), mode = "everything")
print(svm_cm)

# LOGISTIC REGRESSION
# Training
set.seed(123)
lr_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "bayesglm",
    family = "binomial",
    trControl = train_control
)

# Validation
lr_pred <- predict(lr_fit, val)
lr_cm <- confusionMatrix(table(val$V26, lr_pred), mode = "everything")
print(lr_cm)

# K-NEAREST NEIGHBORS
# Training
set.seed(123)
knn_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "knn",
    trControl = train_control
)

# Validation
knn_pred <- predict(knn_fit, val)
knn_cm <- confusionMatrix(table(val$V26, knn_pred), mode = "everything")
print(knn_cm)

# ARTIFICIAL NEURAL NETWORK
# Training
set.seed(123)
nn_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "nnet",
    trControl = train_control
)

# Validation
nn_pred <- predict(nn_fit, val)
nn_cm <- confusionMatrix(table(val$V26, nn_pred), mode = "everything")
print(nn_cm)

# EXTREME GRADIENT BOOSTING
# Training
set.seed(123)
xgb_fit <- train(
    V26 ~.,
    data = train[, -c(1)],
    method = "xgbTree",
    verbosity = 0,
    trControl = train_control
)

# Validation
xgb_pred <- predict(xgb_fit, val)
xgb_cm <- confusionMatrix(table(val$V26, xgb_pred), mode = "everything")
print(xgb_cm)

save.image("cv.RData")