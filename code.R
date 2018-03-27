 ################################################################################
                  # LOADING THE REQUIRED PACKAGES #
 ###############################################################################

library(ggplot2) 
library(readr) 
library(rpart)
library(rattle)
library(ROCR)
library(randomForest)
library(gridExtra)
library(reshape)
library(caTools)
library(e1071)
library(caret)
library(ROCR)
 
 
 ###########################################################################
                       # INPUTING THE DATA FILE #
 ############################################################################

 
df <- read.csv("HR_comma_sep.csv")
# Look at the data
sum(is.na(df))

###########################################################################
                  # PREPROSSING OF THE INPUT DATA#
###########################################################################


sales <- unique(df$sales)
df$sales <- as.numeric(1:10)[match(df$sales, sales)] 
df$salary <- as.numeric(1:3)[match(df$salary, c('low', 'medium', 'high'))]


###########################################################################
                  # VISUALIZATION OF INPUT DATA
############################################################################


p1 <- qplot(satisfaction_level, data=df, geom="histogram", binwidth=0.01)
p2 <- qplot(last_evaluation, data=df, geom="histogram", binwidth=0.01)
p3 <- qplot(number_project, data=df, geom="histogram")
p4 <- qplot(average_montly_hours, data=df, geom="histogram")
p5 <- qplot(time_spend_company, data=df, geom="histogram") 
p6 <- qplot(Work_accident, data=df, geom="histogram")
p7 <- qplot(promotion_last_5years, data=df, geom="histogram")
p8 <- qplot(sales, data=df, geom="histogram")
p9 <- qplot(salary, data=df, geom="histogram")
library(gridExtra)
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, ncol = 3, nrow = 3)




###########################################################################
                     # FEATURE SELECTION #
############################################################################

df$Work_accident <- NULL
df$promotion_last_5years <- NULL


###########################################################################
                       #SPLITTING OF INPUT DATA#
############################################################################

set.seed(300)
df$sp <- sample.split(df$left, SplitRatio=0.8)
train <- subset(df, df$sp==TRUE)
test <- subset(df, df$sp==FALSE)

###########################################################################
                        # LOGISTIC REGRESSION#
############################################################################


# let us first start with logistic regression
# Train the model using the training sets and check score
    model_glm <- glm(left ~ ., data = train, family='binomial')

# Predict Output of test data
    predicted_glm <- predict(model_glm, test, type='response')
    predicted_glm <- ifelse(predicted_glm > 0.5,1,0)

# Confusion matrix of Logistic regression
    table(test$left, predicted_glm)

# Accuracy of model
    mean(predicted_glm==test$left)


###########################################################################
                    # SUPPORT VECTOR MACHINE #
############################################################################
    
# Train the model using the training sets and check score
    model_svm <- svm(left ~ ., data=train)

# Predict Output of test data
    predicted_svm <- predict(model_svm, test)
    predicted_svm <- ifelse(predicted_svm > 0.5,1,0)
    
# Confusion matrix of SVM
    table(test$left, predicted_svm)

# Accuracy of SVM
    mean(predicted_svm==test$left)


###########################################################################
                           # DECISION TREES #
############################################################################


# Let us try decision trees
# Train the model using the training sets and check score    
    model_dt <- rpart(left ~ ., data=train, method="class", minbucket=25)

# View decision tree plot
    fancyRpartPlot(model_dt)

# Predict Output of test data
    predicted_dt <- predict(model_dt, test, type="class")

# Confusion matrix of decision tree
    table(test$left, predicted_dt)
    
# Accuracy of decision tree
    mean(predicted_dt==test$left)
    

###########################################################################
                          #RANDOM FORESTS#
############################################################################


# We shall do random forests with 200 trees
# Train the model using the training sets and check score  
    library(randomForest)
    model_rf <- randomForest(as.factor(left) ~ ., data=train, nsize=20, ntree=200)

# Predict Output of test data
    predicted_rf <- predict(model_rf, test)

# Confusion matrix of random forest
    table(test$left, predicted_rf)

# Accuracy of random forest
    mean(predicted_rf==test$left)


###########################################################################
                        #COMPARSION OF THE METHODS#
##########################################################################
    

# Tuning the parameters increases the accuracy of SVM to 97.53%
# Let us plot the ROC curves for all the models
# Logistic regression
    library(ROCR)
    predict_glm_ROC <- predict(model_glm, test, type="response")
    pred_glm <- prediction(predict_glm_ROC, test$left)
    perf_glm <- performance(pred_glm, "tpr", "fpr")
    
# Decision tree
    predict_dt_ROC <- predict(model_dt, test)
    pred_dt <- prediction(predict_dt_ROC[,2], test$left)
    perf_dt <- performance(pred_dt, "tpr", "fpr")
    
# Random forest
    predict_rf_ROC <- predict(model_rf, test, type="prob")
    pred_rf <- prediction(predict_rf_ROC[,2], test$left)
    perf_rf <- performance(pred_rf, "tpr", "fpr")
    
# SVM
    predict_svm_ROC <- predict(model_svm, test, type="response")
    pred_svm <- prediction(predict_svm_ROC, test$left)
    perf_svm <- performance(pred_svm, "tpr", "fpr")
    
# Area under the ROC curves
    auc_glm <- performance(pred_glm,"auc")
    auc_glm <- round(as.numeric(auc_glm@y.values),3)
    auc_dt <- performance(pred_dt,"auc")
    auc_dt <- round(as.numeric(auc_dt@y.values),3)
    auc_rf <- performance(pred_rf,"auc")
    auc_rf <- round(as.numeric(auc_rf@y.values),3)
    auc_svm <- performance(pred_svm,"auc")
    auc_svm <- round(as.numeric(auc_svm@y.values),3)
    print(paste('AUC of Logistic Regression:',auc_glm))
    print(paste('AUC of Decision Tree:',auc_dt))
    print(paste('AUC of Random Forest:',auc_rf))
    print(paste('AUC of Support Vector Machine:',auc_svm))
    
    
# Plotting the three curves
    plot(perf_glm, main = "ROC curves for the models", col='blue')
    plot(perf_dt,add=TRUE, col='red')
    plot(perf_rf, add=TRUE, col='green3')
    plot(perf_svm, add=TRUE, col='darkmagenta')
    legend('bottom', c("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"), fill = c('blue','red','green3','darkmagenta'), bty='n')
