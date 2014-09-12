setwd("I:\\data")
library(data.table)
train <- fread("train.csv",select=c(1:15))
head(train)
write.table(train,"train_num.csv",sep=",", row.names=F, col.names=T)
str(train)
# rm("train")

