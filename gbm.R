setwd("I:\\data")
library(data.table)
train <- fread("train.csv",select=c(1:15))
head(train)
write.table(train,"train_num.csv",sep=",", row.names=F, col.names=T)
str(train)
# rm("train")
?read.csv

library(caret)
sum(is.na(train))
mean(is.na(train))
summary(train)
gc()
train <- fread("train_num.csv")
train <- na.omit(train)
# index <- is.na(train)
# table(index)
# train <- train[-index,]
write.table(train,"train_num_na.csv",sep=",", row.names=F, col.names=T)
rm(train)

# data cleansing
#################
train <- read.csv("train_num_na.csv")
summary(train)
head(train)
train <- train[,-1]
train$Label <- as.factor(train$Label)
boxplot(train)

#covariate creation
nearZeroVar(train,saveMetrics = T)
train.pca <- preProcess(train[,2:14], method='pca', pcaComp=2)

# model
library(doParallel)
library(caret)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
fit1 <- train(Label~., method="rf",data=train)

    Grid <- expand.grid(n.trees=c(500),interaction.depth=c(22),shrinkage=.2)
    fitControl <- trainControl(method="none", allowParallel=T, classProbs=T)
fit2 <- train(Label~., method="gbm", data=train, trControl=fitControl, verbose=T,tuneGrid=Grid, metric="ROC")

fit3 <- train(Label~., method="nb", data=train)
stopCluster(cl)

# load test data
test <- fread("test.csv", select=c(1:15))
write.table(test,"test_num.csv",sep=",", row.names=F, col.names=T)

# predict
gc()
pred1 <- predict(fit1, test)
pred2 <- predict(fit2, test)
pred3 <- predict(fit3, test)
# ensembling-models
data(pred1,pred2,pred3,train)
combFit<-train(Label~.,method="gam", train)