setwd('/Users/ivan/Work_directory/Predict-click-through-rates-on-display-ads/')
train <- 'testSet.txt'
test <- 'test.csv'
D <- 2^20
alpha <- .1
w <- rep.int(0, D)
n <- rep.int(0, D)
loss <- 0.
col_num <- 3

# test logloss of predictions and true values
logloss <- function(p,y) {
    p <- max(min(p, 1 - 10^-13), 10^-13)
    res <- ifelse(y==1, -log(p), -log(1 - p))
    res
}

# extract one record from database	    
get_data <- function(row, data){
	r <- read.table(data, skip=row-1, nrows=1, sep='\t', row.names=row)
    r
}

# get possibilities of records
get_p <- function(x, w){
	wTx <- 0
	for (i2 in x) {
		wTx <- wTx + w[i2] * 1.}
    p <- 1/(1 + exp(-max(min(wTx, 20), -20)))
    p 
}

# update the weights according to results
update_w <- function(w, n, x, p, y){
	for (i in x){
        w[i] <- w[i] - ((p - y) * alpha / (sqrt(n[i]) + 1))
        n[i] <- n[i] + 1
	}
    w
    n
}

# main steps for modeling
for i in 1:46000000{
	row <- get_data(i, train,2)
	y <- row[2]
	row <- row[-c(1,2)]
	p <- get_p(row, w)
	loss <- loss + logloss(p,y)
	if(i%1000000 == 0){
		'logloss: '&loss/i&'. number of row: 'i
	}
	updates <- update_w(w, n, x, p, y)
	w <- updates[1]
	n <- updates[2]
}


get_x <- function(row, D){
    x <- c(0)
    for (j in row){
        index <- as.integer(j) %% D
        x <- c(x, index)
    }
}





