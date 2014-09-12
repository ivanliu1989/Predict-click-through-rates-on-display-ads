setwd("I:\\data")
library(data.table)
train <- fread("train.csv",select=c(1:15))
head(train)
write.table(train,"train_num.csv",sep=",", row.names=F, col.names=T)
str(train)
# rm("train")

library(caret)
sum(is.na(train))
mean(is.na(train))
summary(train)
gc()
train <- read.csv("train_num.csv")
index <- is.na(train)
table(index)
train <- train[-index,]
write.table(train,"train_num_na.csv",sep=",", row.names=F, col.names=T)
rm(train)

# imputation
train <- read.csv("train_num_na.csv")
train <- train[,-1]