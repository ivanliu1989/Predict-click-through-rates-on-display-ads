setwd("I:\\data")
# library(data.table)
#     # train <- fread("train.csv",select=c(1:15))
# train <- fread("data/train.csv",select=c(1,16:20))
# train <- fread("data/train.csv",select=c(1,21:25))
# train <- fread("data/train.csv",select=c(1,26:30))
# train <- fread("data/train.csv",select=c(1,31:35))
# train <- fread("data/train.csv",select=c(1,36:38))
# train <- fread("data/train.csv",select=c(1,39:41))
#     # head(train)
# train <- na.omit(train)
# train$C1 <- as.integer(train$C1)
# train$C2 <- as.integer(train$C2)
# train$C3 <- as.integer(train$C3)
# train$C4 <- as.integer(train$C4)
# train$C5 <- as.integer(train$C5)
# train$C6 <- as.integer(train$C6)
# train$C7 <- as.integer(train$C7)
# train$C8 <- as.integer(train$C8)
# train$C9 <- as.integer(train$C9)
# train$C10 <- as.integer(train$C10)
# train$C11 <- as.integer(train$C11)
# train$C12 <- as.integer(train$C12)
# train$C13 <- as.integer(train$C13)
# train$C14 <- as.integer(train$C14)
# train$C15 <- as.integer(train$C15)
# train$C16 <- as.integer(train$C16)
# train$C17 <- as.integer(train$C17)
# train$C18 <- as.integer(train$C18)
# train$C19 <- as.integer(train$C19)
# train$C20 <- as.integer(train$C20)
# train$C21 <- as.integer(train$C21)
# train$C22 <- as.integer(train$C22)
# train$C23 <- as.integer(train$C23)
# train$C24 <- as.integer(train$C24)
# train$C25 <- as.integer(train$C25)
# train$C26 <- as.integer(train$C26)
    # write.table(train,"train_num.csv",sep=",", row.names=F, col.names=T)
# write.table(train,"train_cat_1.csv",sep=",", row.names=F, col.names=T)
# write.table(train,"train_cat_2.csv",sep=",", row.names=F, col.names=T)
# write.table(train,"train_cat_3.csv",sep=",", row.names=F, col.names=T)
# write.table(train,"train_cat_4.csv",sep=",", row.names=F, col.names=T)
# write.table(train,"train_cat_5.csv",sep=",", row.names=F, col.names=T)
# write.table(train,"train_cat_6.csv",sep=",", row.names=F, col.names=T)
# train <- read.csv("train_cat_6.csv", sep=",")
    # str(train)
# rm("train")
    # ?read.csv
    # 
    # library(caret)
    # sum(is.na(train))
    # mean(is.na(train))
    # summary(train)
    # gc()
    # train <- fread("train_num.csv")
    # train <- na.omit(train)
# index <- is.na(train)
# table(index)
# train <- train[-index,]
    # write.table(train,"train_num_na.csv",sep=",", row.names=F, col.names=T)
    # rm(train)

# data cleansing
#################
train <- read.csv("train_num_na.csv")
train <- train[,-1]
head(train)
train[which(train$Label==1),1] <- "Yes"
train[which(train$Label==0),1] <- "No"
train$Label <- as.factor(train$Label)
write.table(train,"train_num_na_yesno.csv",sep=",", row.names=F, col.names=T)


#covariate creation
    # nearZeroVar(train,saveMetrics = T)
    # train.pca <- preProcess(train[,2:14], method='pca', pcaComp=2)

# model
    # library(doParallel)
library(caret)
    # cl <- makePSOCKcluster(4)
    # registerDoParallel(cl)
# fit1 <- train(Label~., method="rf",data=train)

    Grid <- expand.grid(n.trees=c(800),interaction.depth=c(22),shrinkage=.1)
    fitControl <- trainControl(method="cv",2, allowParallel=T, classProbs=T)
fit3 <- train(Label~., method="gbm", data=train, trControl=fitControl, 
              verbose=T,tuneGrid=Grid, metric="ROC")
pred2 <- predict(fit2, train, type="prob")
confusionMatrix(pred2, train$Label)
rm(pred2)

# fit3 <- train(Label~., method="glmnet",family="binomial",classProbs=T, data=train,verbose=T)
# fit3 <- train(Label~., method="glmnet",classProbs=T, data=train,verbose=T)

# fit 4
ctrl <- trainControl(method = "cv",
                     number=2,
                     classProbs = TRUE,
                     allowParallel = TRUE,
                     summaryFunction = twoClassSummary)

set.seed(888)
rfFit <- train(Label~.,
               data=train,
               method = "rf",
#                tuneGrid = expand.grid(.mtry = 4),
               ntrees=500,
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)


pred <- predict.train(rfFit, newdata = test, type = "prob") 


# stopCluster(cl)

# load test data
    # test <- fread("test.csv", select=c(1:14))
    # write.table(test,"test_num.csv",sep=",", row.names=F, col.names=T)
    # test <- read.csv("test_num.csv")
test <- read.csv("test_num_impute.csv")
# test data imputation
    # pre<-preProcess(test, method='medianImpute')
    # test_impute <- predict(pre, test)

# predict
gc()
    # pred1 <- predict(fit1, test)
pred2 <- predict(fit2,type="prob", test)
head(test)
pred2 <- plogis(pred2)
    # pred3 <- predict(fit3, test)
# ensembling-models
    # data(pred1,pred2,pred3,train)
    # combFit<-train(Label~.,method="gam", train)

# output
fit2.submit <- data.frame(test$Id, pred2$Yes)
colnames(fit2.submit)<- c("Id","Predicted")
write.table(fit2.submit,"submit1_gbm_num_nona_impute.csv", row.names=F, sep=',')
