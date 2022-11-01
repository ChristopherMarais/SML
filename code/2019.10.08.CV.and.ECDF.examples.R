# SML 2019, 08-Oct-2019 

# Example 1: wrong and right ways to do cross-validation
# Example 2: ECDF illustration

# Example 1: wrong and right ways to do cross-validation
# Part 0: setup
set.seed(0)
K = 10;  # number of folds in the CV
n = 100;  # make this a multiple of K for convenience
p = 5000; # compare with smaller p, e.g., p= 50 or 100
Y = rep(c(0,1),n/2)
Y = sample(Y, size=n)
folds.ids = rep(seq(1,K),n/K)
folds.ids = sample(folds.ids)
X = matrix(rnorm(n*p),nrow=n); # features are iid normal
# X = matrix(runif(n*p),nrow=n) # features are iid uniform

library(MASS) # needed for LDA

# Part 1: CV done wrong
# prescreen: 
p0 = 50;
cor.out = rep(NA, p)
for (i in 1:p) {cor.out[i] = cor(Y, X[,i])}
plot(cor.out)
out1 = sort(abs(cor.out), decreasing=TRUE, index.return=TRUE); # sort abs values of correlations from largest to smallest
ii = out1$ix[1:p0]; # indices of p0 largest correlations (in abs. value)
# cor.out[ii]
df0 = data.frame(X=X[,ii],Y=Y)
df0$Y = Y;
#
# prescreening function; returns indices/ids of variables in X0 with highest (abs) correlations with the response Y0
myPrescreen <- function(Y0, X0, p0) {
  cor.out = rep(NA, p)
  for (i in 1:p) {cor.out[i] = cor(Y0, X0[,i])}
  out1 = sort(abs(cor.out), decreasing=TRUE, index.return=TRUE)
  ii = out1$ix[1:p0]
  df0 = data.frame(X=X0[,ii],Y=Y0)
  ii
}
jj = myPrescreen(Y, X, p0);


#
out.glm = rep(-1,K); out.lda = out.glm; # allocate storage for results
for (i in 1:K){
  valid = (folds.ids == i);
  # train = (folds.ids != i);
  fit1 <- glm(Y ~ ., family=binomial, data=df0, subset=!valid); 
  pred1 = predict(fit1, df0[valid,], type="response")
  out.glm[i] = mean(abs(Y[valid]-round(pred1)))
  #
  lda1 <- lda(Y ~ ., data=df0, subset=!valid);
  pred2 <- predict(lda1, df0[valid,])$class
  pred3 = as.integer(pred2)-1; # examine class of pred 2; convert to numerical;
  out.lda[i] = mean(abs(Y[valid]-pred3))
  #
}
xx = cbind(out.glm, out.lda); print(xx)
apply(xx, 2, mean)


# Part 2: CV done right
out.glm = rep(-1,K); out.lda = out.glm; 
iiM = matrix(rep(0,p0*K),nrow=K); # also record which variables were chosen in each training fold
for (i in 1:K){
  valid = (folds.ids == i);
  ii  = myPrescreen(Y[!valid], X[!valid,],p0);
  dft = data.frame(Y = Y[!valid], X = X[!valid,ii])
  dfv = data.frame(Y = Y[valid], X = X[valid,ii])
  #
  fit1 <- glm(Y ~ ., family=binomial, data=dft); 
  pred1 = predict(fit1, dfv, type="response")
  out.glm[i] = mean(abs(Y[valid]-round(pred1)))
  #
  lda1 <- lda(Y ~ ., data=dft);
  pred2 <- predict(lda1, dfv)$class
  pred3 = as.integer(pred2)-1
  out.lda[i] = mean(abs(Y[valid]-pred3))
  #
  iiM[i,] = ii;
}
xx2 = cbind(out.glm, out.lda); print(xx2)
apply(xx2, 2, mean)
#
# visualize common predictors using set intersection:
commonM = matrix(rep(0,K^2),nrow=K)
rc.names = paste("F",seq(1,K),sep="")
colnames(commonM) = rc.names; rownames(commonM) = rc.names
for (i1 in 1:K) {
  for (i2 in i1:K){
    commonM[i1,i2] = length(intersect(iiM[i1,],iiM[i2,]))
  }
}
print(commonM)



# ECDF is a consistent estimator of the CDF

myf <- function(n,rseed=0){
  set.seed(rseed)
  x = rnorm(n)
  xgrid = seq(-3,3,length.out = 1001)
  Fxgrid = pnorm(xgrid)
  Fhat = ecdf(x)
  plot(xgrid,Fxgrid,type="l",lwd=2)
  Fhat.grid = Fhat(xgrid)
  points(xgrid,Fhat.grid,col="red", type="l",lwd=2,lty=2)
  points(x, rep(0,length(x)), pch="+", col="blue",cex=2)
  title(paste("n=",n,sep=""))
}
myf(5,2)
myf(100,2)
myf(1000,2)
myf(10^4,2)
