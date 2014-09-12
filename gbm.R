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
fit1 <- train(Label~., method="rf",data=train, prox=T)
fit2 <- train(Label~., method="gbm", data=train)
fit3 <- train(Label~., method="glm", data=train)
fit4 <- train(Label~., method="bstTree", data=train)
fit5 <- train(Label~., method="glmboost", data=train)
fit6 <- train(Label~., method="nb", data=train)

pred1 <- predict(fit1, test)
pred2 <- predict(fit2, test)
pred3 <- predict(fit3, test)
pred4 <- predict(fit4, test)
pred5 <- predict(fit5, test)
pred6 <- predict(fit6, test)

data(pred1,pred2,pred3,pred4,pred5,pred6,train)
combFit<-train(Label~.,method="gam", train)