train <- 'train.csv'
test <- 'test.csv'
D <- 2^20
alpha <- .1
w <- rep(0, n=D)
n <- rep(0, n=D)
loss <- 0

logloss <- function(p,y){
	p <- max(min(p, 1 - 10e-12), 10e-12)
    return ifelse(y==1, -log(p), -log(1. - p))
}
get_data <- function(row, data){
	r <- read.table(data, colclass=c(rep('Integer', n=41)), skip=row-1, nrows=row, sep=',', colnames=F)
	return r
}

get_p <- function(x, w){
	wTx <- 0
	for i in 1:x
		wTx <- wTx + w[i] * 1.
    p <- 1/(1 + exp(-max(min(wTx, 20), -20)))
    return p 
}

update_w <- function(w, n, x, p, y){
	for i in 1:x
        w[i] <- w[i] - ((p - y) * alpha / (sqrt(n[i]) + 1))
        n[i] <- n[i] + 1
    return w, n
}






for i in 1:46000000{
	

}