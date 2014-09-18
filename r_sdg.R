setwd('/Users/ivan/Work_directory/Predict-click-through-rates-on-display-ads/')
train <- 'testSet.txt'
test <- 'test.csv'
D <- 2^20
alpha <- .1
w <- rep.int(0, D)
n <- rep.int(0, D)
loss <- 0.
col_num <- 3

logloss <- function(p,y) {
    p <- max(min(p, 1 - 10^-13), 10^-13)
    res <- ifelse(y==1, -log(p), -log(1 - p))
    res
}
	    
get_data <- function(row, data, col){
	r <- read.table(data, 
                    skip=row-1, nrows=row, sep='\t', col.names=row)
	r
}

get_p <- function(x, w){
	wTx <- 0
	for i in 1:x
		wTx <- wTx + w[i] * 1.
    p <- 1/(1 + exp(-max(min(wTx, 20), -20)))
    p 
}

update_w <- function(w, n, x, p, y){
	for i in 1:x
        w[i] <- w[i] - ((p - y) * alpha / (sqrt(n[i]) + 1))
        n[i] <- n[i] + 1
    return w, n
}

for i in 1:46000000{
	x <- get_data(i, train)
	y <- x[2]
	x <- x[-c(1,2)]
	p <- get_p(x, w)
	loss <- loss + logloss(p,y)
	if(i%1000000 == 0){
		'logloss: '&loss/i&'. number of row: 'i
	}
	updates <- update_w(w, n, x, p, y)
	w <- updates[1]
	n <- updates[2]
}

w