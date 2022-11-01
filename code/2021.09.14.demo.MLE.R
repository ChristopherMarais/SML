

# stuff to show in R:

# 1. computation of moments
# dnorm(x, mean = 0, sd = 1, log = FALSE)
gk <- function(x) {x^4 * dnorm(x)}  # standard normal
mk = integrate(gk, -10,10); mk
mk = integrate(gk, -Inf, Inf); mk

mut = 17
gk <- function(x) {x * dnorm(x, mean=mut, sd=1)}  # standard normal
mk = integrate(gk, -20,30); mk
mk = integrate(gk, -Inf, Inf); mk

myf1 <- function(x) {dnorm(x)}
c = 50;
myf2 <- function(x) {0.5*(dnorm(x)+dnorm(x,mean=c))}
myg2 <- function() {
  i1 = integrate(myf2, -100,100); print(i1)
  i2 = integrate(myf2, -Inf, Inf)
  cbind(i1[1],i2[1])
  
}
myg2()

# Expectations of functions of rvs: from pretest
# Ybar = X/n is an estimator of E(Y). X ~ Gamma(shape=n, rate=1).
n=5;
f1 <- function(x) {x/n * dgamma(x,shape=n,rate=1)}
f2 <- function(x) {(x/n)^2 * dgamma(x,shape=n,rate=1)}
M1 = integrate(f1,0,100); print(M1)
# M1 = integrate(f1,0,Inf); print(M1)
# M1 = integrate(f1,0,10^5); print(M1)
M2 = integrate(f2,0,100); print(M2)
V = M2$value - (M1$value)^2;
B = M1$value - 1; # bias
print(c(B, V))
#
# Exercise: modify the script above to compute bias and variance for lambdaHat = 1/Ybar

# 2. maximum likelihood and optimization
# 2.1. example using plotting (exact solution) aka grid search
binlikf <- function(p) {dbinom(67,size=100, prob=p)}
pgrid = seq(0,100,1)/100
fv = binlikf(pgrid)
plot(pgrid,fv)
out = max(fv)
ind = (fv == out)
pgrid[ind]
# alternatively, use 
ii = which(ind); pgrid[ii]
# alt, write a simple function
myArgMax <- function(x,fv) {
  out = max(fv)
  ind = (fv == out)
  x[ind]
}

# 2.2. change grid
pgrid = seq(0,10,1)/10
fv = binlikf(pgrid)
plot(pgrid,fv)
myArgMax(pgrid,fv)

# 2.3. use optimization
gopt <- function(p) {-binlikf(p)} # objective function
out = optim(0.5, gopt); out  # check out documentation
out2 = optim(0.5, gopt, method="BFGS"); out2 
out3 = optim(0.5, gopt, method="L-BFGS-B", lower=10^(-5), upper=1-10^(-5)); out3 


# 2.4. using log-likelihood and random successes:
set.seed(0) # set seed for reproducibility of results
bernv <- runif(100) < 0.7
bernv <- as.numeric(bernv)
bernL  <- function(p) {prod(dbinom(bernv, size=1, prob=p, log=FALSE))}
bernLL <- function(p) {sum(dbinom(bernv, size=1, prob=p, log=TRUE))}
# pgrid = seq(1,99,1)/100
outL  = rep(NA, length(pgrid))
for (i in 1:length(pgrid)) {outL[i] = bernL(pgrid[i])}
plot(pgrid, outL)
myArgMax(pgrid,outL)

outLL  = rep(NA, length(pgrid))
for (i in 1:length(pgrid)) {outLL[i] = bernLL(pgrid[i])}
plot(pgrid, outLL)
myArgMax(pgrid, outLL)

# use optimization:
obj1 <- function(p) {(-1)*bernL(p)}
obj2 <- function(p) {(-1)*bernLL(p)}
out1 = optim(0.5, obj1); out1

out2 = optim(0.5, obj2); out2

# 2.5 Exercise: Poisson ML estimation
# 1. simulate 10 Poisson(lambda_TRUE) rvs, lambda_TRUE = 17
set.seed(0); n = 10; y = rpois(n,17)

pLL  <- function(a) {sum(dpois(y, lambda=a, log=TRUE))}# loglik
obj3 <- function(a) {-pLL(a)}
out = optim(1, obj3)
out
mean(y)

# SML class - omit
# suppose the data are Poisson with pop. mean equal to beta^3
# what is the MLE of beta?
pLL2  <- function(a) {sum(dpois(y, lambda=a^3, log=TRUE))}# loglik
obj4 <- function(a) {-pLL2(a)}
out = optim(1, obj4)
out
# compare mean(y) with out[1]^3
# "invariance property of MLE"


# 2.6 Exercise: Normal(mu,sigma) ML estimation
# 1. simulate 27 Normal(mut,sdt) rvs; mut = 17; sdt=4 - true values
set.seed(0)
n = 27; mut = 17; sdt = 4;
y = rnorm(n = n, mean = mut, sd = sdt)

# 2. find the MLE of (mu, sigma)
negLL <- function(a) {(-1)*sum(dnorm(y, mean=a[1], sd = a[2], log=TRUE))}
negLL(c(0,1))

# updated 17-Sep-2020: why does it fail to converge?
out = optim(c(0,1), negLL); out 
out = optim(c(0,1), negLL,method="BFGS"); out # fails to converge
out = optim(c(mut,sdt), negLL,method="BFGS"); out # converges but from a different starting point
out = optim(c(0,1), negLL,method="L-BFGS-B",lower=c(-10^4,10^(-4))); out 
# explicitly set up the lower bound for parameters; converges from c(0,1)
#
mean(y) # MLE for mu (analytical)
sqrt(var(y)*(n-1)/n)  # MLE for sigma (analytical); var(y) is the sample variance of Y

# Updated/created on 17-Sep-2020: created due to popular demand from students in class.
# Exercise (optional exercise 6 for hw1): MLE by numerical optimization for a bivariate normal model
# simulate the data as follows from a bivariate normal pdf as follows:
set.seed(0)
n = 20; muv = c(0,17); sv = c(1,2); rho = 0.5; #  muv - vector of means; sv - vector of std. devs, rho is the correlation coeff.
Sig = matrix(c(sv[1]^2, rho*prod(sv), rho*prod(sv), sv[2]^2),nrow=2)
R = chol(Sig); # Sig-t(R)%*% R
Z = matrix(rnorm(n*2),ncol=n)
Y = t(t(R)%*%Z); # cov(Y) - Sig; (increase n to experiment see that cov(Y) -> Sig)
Y[,1] = Y[,1]+muv[1]; Y[,2]=Y[,2]+muv[2]; # rows of Y are the iid data from the bivariate normal model; 
#
# estimate theta = c(muv, sv, rho) numerically by MLE
# in order to do so, first implement the logarithm of the bivariate normal pdf by hand;


# Bayesian inference - finding posterior mode  # optional
# Exercise: modify example 2.4 to use left-skewed and right-skewed beta priors
# using dbeta function in R. Use n=10 rather than n=100 observations. Compare your results with MLE estimation.

# 2022-10-06: logistic regression quirks 
set.seed(0); 
n = 40; 
u = runif(n);
x = c(seq(-1,-0.2,length.out=n/2), seq(0.2,1,length.out=n/2))
c=2^1; # c = 0,1,2,4,8,16
px = exp(c*x)/(1+exp(c*x)); # true prob of success
y = (u<px); # generate the responses so that y[i] is Bern(p=px[i])
plot(x,y); points(x,px,pch="+",col="red") # generate data & visualize
fit = glm(y~x,family=binomial); summary(fit)

# same idea, but for a categorical binary predictor
myF = as.factor(x > 0)
fit = glm(y~myF,family=binomial); summary(fit)


# 2020-10-12: ROC plot illustration
library(ROCR)
rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

set.seed(20); 
n = 10;
x = c(seq(-1,-0.2,length.out=n/2), seq(0.2,1,length.out=n/2))
c=2; # c = 0,1,2,4,8,16
px = exp(c*x)/(1+exp(c*x)); # true prob of success
y = (runif(n)<px); sum(y)
y[4] = 1; # making the fit stable - ad hoc; repeat without and with this line
plot(x,y); points(x,px,pch="+",col="red") # generate data & visualize
fit = glm(y~x,family=binomial); summary(fit)
#
# ROC plot to training data
est.p = fit$fitted.values;
rocplot(est.p, y)
M = cbind(est.p,y); M
#
P = sum(y); N = n-sum(y); c(P,N)
