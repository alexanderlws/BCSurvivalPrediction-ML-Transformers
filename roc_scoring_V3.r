# ROC Curves + Scoring
library(caret)
library(pROC)
library(RColorBrewer)

load("cv.RData")

# ROC-TRAINING
colour <- brewer.pal(n = 7, name = "Dark2")
dev.new()

set.seed(123)
par(pty = "s")
roc_RF <- roc(
       response = val$V26,
       predictor = as.numeric(rf_pred),
       plot = TRUE, main = "AUC-ROC CURVE",
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.05,
       col = colour[1],
       lwd = 2, legacy.axes = TRUE
)

roc_DT <- roc(
       response = val$V26,
       predictor = as.numeric(dt_pred), 
       plot = TRUE, add = TRUE,
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.10,
       col = colour[2],
       lwd = 3, legacy.axes = TRUE
)

roc_SVM <- roc(
       response = val$V26,
       predictor = as.numeric(svm_pred),
       plot = TRUE, add = TRUE,
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.15,
       col = colour[3],
       lwd = 3, legacy.axes = TRUE
)

roc_LR <- roc(
       response = val$V26,
       predictor = as.numeric(as.factor(lr_pred)),
       plot = TRUE, add = TRUE,
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.20,
       col = colour[4],
       lwd = 3, legacy.axes = TRUE
)

roc_KNN <- roc(
       response = val$V26,
       predictor = as.numeric(knn_pred),
       plot = TRUE, add = TRUE,
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.25,
       col = colour[5],
       lwd = 3, legacy.axes = TRUE
)

roc_NN <- roc(
       response = val$V26,
       predictor = as.numeric(as.factor(nn_pred)),
       plot = TRUE, add = TRUE,
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.30,
       col = colour[6],
       lwd = 3, legacy.axes = TRUE
)

roc_XGB <- roc(
       response = val$V26,
       predictor = as.numeric(xgb_pred),
       plot = TRUE, add = TRUE,
       print.auc = TRUE, print.auc.x = 0.3, print.auc.y = 0.35,
       col = colour[7],
       lwd = 3, legaxy.axes = TRUE
)

legend(x = -0.1, y = 0.6, legend = c("RF", "DT", "SVM", "LR", "KNN", "NN", "XGB"), col = colour[1:7], lwd = 4, xpd = TRUE)

# PLOT ACCURACY, SPECIFICITY, SENSITIVITY - TRAINING
#cm_list <- list("rf_cm", "dt_cm", "svm_cm", "lr_cm", "knn_cm", "nn_cm", "xgb_cm")
#mt <- data.frame(alg = character(), acc = numeric(), sen = numeric(), spe = numeric())
#for (i in cm_list)
#{
#       a <- as.matrix(eval(parse(text = i)), what = "overall")
#       b <- as.matrix(eval(parse(text = i)), what = "classes")
#       c <- data.frame(alg = i, acc = a[1], sen = b[1], spe = b[2])
#       mt <- rbind(mt, c)
#}

#mt$alg <- c("RF", "DT", "SVM", "LR", "KNN", "NN", "XGB")

#mt_re <- as.matrix(mt[, -1])
#colnames(mt_re) <- c("ACCURACY", "SENSITIVITY", "SPECIFICITY")
#rownames(mt_re) <- mt$alg

#trans_mt <- t(mt_re)
#View(trans_mt)