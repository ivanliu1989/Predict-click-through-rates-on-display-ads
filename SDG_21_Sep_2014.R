setwd("C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\Predict-click-through-rates-on-display-ads\\local")
# basic parameters
con <- file('train.csv','r')
D <- 2^27
alpha <- .145

# logit loss calculation
logloss <- function(p,y){
    epsilon <- 10 ^ -15
    p <- max(min(p, 1-epsilon), epsilon)
    ll <- y*log(p) + (1-y)*log((1-p))
    ll <- ll * -1/1
    ll
}

# prediction
get_p<- function(x,w){
    wTx <- 0
    for (i in 1:length(x)){
        wTx <- wTx + w[i] * 1
    }
    sigmoid <- 1/(1+exp(-max(min(wTx,20)-20))) 
    sigmoid
}

# update weights
update_w <-function (w, n, x, p, y){
    for (i in 1:length(x)){
        lr <- alpha / (sqrt(n[i])+1)
        gradient <- (p-y) 
        w[i] <- w[i] - gradient * lr
        n[i] <- n[i] + 1
    }    
    c(w,n)
}

# basic parameters
# w <- rep(0, D)
w <- rep(1, length(x))
# n <- rep(0, D)
n <- rep(0, length(x))
loss <- 0
##################start modeling - parameters setup ########################
train_label <- readLines(con, n=1)
for (i in 1:10) {
row <- readLines(con,n=1)
train_row <- strsplit(row, ',')
y <- as.integer(train_row[[1]][2])
x <- c()
for (k in 3:40){
    x <- c(x, train_row[[1]][k])
}
################# Modeling #################################################
p <- get_p(x,w)
loss <- loss + logloss(p,y)
if (i %% 100000 == 0){
    print(loss)
} 
upd <- update_w(w,n,x,p,y)
w <- upd[1]
n <- upd[2]
print(loss)
break
}
