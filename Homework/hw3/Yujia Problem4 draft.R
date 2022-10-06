n = 30; set.seed(0);
X = runif(n)*4-2;
Y = 1 + 3*X + rnorm(n);
fit1 = lm(Y ~ X); summary(fit1)
#4.1
myFullObj <- function(b,sig) {
  miu <- b[1]+mean(X)*b[2]
  (-1)*sum(dnorm(Y, mean=miu, sd = sig, log=TRUE))
  
}
#4.2
sigKnown = 2; myObj1 <- function(b) {myFullObj(b,sigKnown)};
optim(par=c(1,3),myObj1,method="BFGS")
#4.3
optim(c(1,3,1), myFullObj,method="L-BFGS-B",lower=c(-10^4,Inf))
