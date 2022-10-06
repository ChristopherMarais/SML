#3.1

n = 30; set.seed(0); p = 3;
X = matrix(runif(n*p),nrow=n)*2-1;
b = seq(1,p,by=1);
Y = X%*%b + rnorm(n);

L1 <- vector(,n)
L2 <- vector(,n)

#OLS model without intercept
myOLS <- function(Y, X) {
  XprimeX <- t(X) %*% X                #X'X is pxp matrix
  XprimeY <- t(X) %*% Y                #X'Y is px1 vector
  invXX <- solve(XprimeX)              #(X'X)^(-1) is pxp inverse of X'X
  betahat <- invXX %*% XprimeY         #betahat is px1 matrix 
  L1 <- betahat
  YprimeY <- t(Y) %*% Y                #Y'Y = sum(Y_i^2)  
  s2 <- (YprimeY - (t(betahat) %*% XprimeY)) / (n-p)  #s^2 is unbiased estimate of sigma^2
  V_betahat <- (s2[1,1] * invXX)         #var-cov matrix of bhat
  SE_betahat <- (sqrt(diag(V_betahat)))  #SE's of bhat
  L2 <- SE_betahat
  
  L<-cbind(L1,L2)
  colnames(L) <- c("L[1]", "L[2]")
  print(L)
}
myOLS(Y,X)
fit0 = lm(Y ~ -1 + X); summary(fit0) # regression without an intercept

#OLS model with intercept
myOLSintercept <- function(Y, X) {
  Xnew<-matrix(c(rep(1,n),X),ncol=p+1)
  XprimeX <- t(Xnew) %*% Xnew               
  XprimeY <- t(Xnew) %*% Y             
  invXX <- solve(XprimeX)             
  betahat <- invXX %*% XprimeY       
  L1 <- betahat
  YprimeY <- t(Y) %*% Y                 
  s2 <- (YprimeY - (t(betahat) %*% XprimeY)) / (n-p)  
  V_betahat <- (s2[1,1] * invXX)       
  SE_betahat <- (sqrt(diag(V_betahat))) 
  L2 <- SE_betahat
  
  L<-cbind(L1,L2)
  colnames(L) <- c("L[1]", "L[2]")
  print(L)
}
myOLSintercept(Y,X)
fit1 = lm(Y ~ X); summary(fit1); # regression with an intercept

#3.2
n = 30; set.seed(0);
X = runif(n)*4-2; # X is uniformly distributed on [-2,2]
Y = 1 + 3*X -2*X^2 + 1*X^3 + rnorm(n);

#Polynomial moddel with intercept
myPolyReg1 <- function(Y, X, deg) {
  Xnewnew<-matrix(c(rep(1,n),X,X^2,X^3),ncol=deg+1)
  XprimeX <- t(Xnewnew) %*% Xnewnew                
  XprimeY <- t(Xnewnew) %*% Y                
  invXX <- solve(XprimeX)             
  betahat <- invXX %*% XprimeY         
  L1 <- betahat
  YprimeY <- t(Y) %*% Y                 
  s2 <- (YprimeY - (t(betahat) %*% XprimeY)) / (n-p)  
  V_betahat <- (s2[1,1] * invXX)       
  SE_betahat <- (sqrt(diag(V_betahat))) 
  L2 <- SE_betahat
  
  L<-cbind(L1,L2)
  colnames(L) <- c("L[1]", "L[2]")
  print(L)
}
myPolyReg1(Y,X,deg = 3)
fit0 = lm(Y ~ X + I(X^2) + I(X^3)); summary(fit0)

#3.3
n = 30; set.seed(0);
XF = rep(c("A","B","C"),each=10)
Y = rnorm(n) + rep(c(1,2,3),each=10)

#ANOVA model without intercept
X1 <- matrix(rep(0,30),ncol = 1)
X2 <- matrix(rep(0,30),ncol = 1)
X3 <- matrix(rep(0,30),ncol = 1)
myAnova1 <- function(Y, XF) {
  for (i in c(1:n)) {

  if (XF[i]=="A") {
    X1[i] <- 1
  }
    if (XF[i]=="B") {
      X2[i] <- 1
    }
    if (XF[i]=="C") {
      X3[i] <- 1
    }
  }
  Xnewnewnew<- matrix(c(X1,X2,X3),ncol = 3)
  XprimeX <- t(Xnewnewnew) %*% Xnewnewnew               
  XprimeY <- t(Xnewnewnew) %*% Y                
  invXX <- solve(XprimeX)              
  betahat <- invXX %*% XprimeY        
  L1 <- betahat
  YprimeY <- t(Y) %*% Y                 
  s2 <- (YprimeY - (t(betahat) %*% XprimeY)) / (n-p)  
  V_betahat <- (s2[1,1] * invXX)       
  SE_betahat <- (sqrt(diag(V_betahat))) 
  L2 <- SE_betahat
  
  L<-cbind(L1,L2)
  colnames(L) <- c("L[1]", "L[2]")
  print(L)
}
myAnova1(Y, XF)
fit0 = lm(Y ~ -1 + XF); summary(fit0) # without an intercept

#ANOVA model with intercept
X1 <- matrix(rep(0,30),ncol = 1)
X2 <- matrix(rep(0,30),ncol = 1)
X3 <- matrix(rep(0,30),ncol = 1)
myAnova1intercept <- function(Y, XF) {
  for (i in c(1:n)) {
    
    if (XF[i]=="A") {
      X1[i] <- 1
    }
    if (XF[i]=="B") {
      X2[i] <- 1
    }
    if (XF[i]=="C") {
      X3[i] <- 1
    }
  }
  Xnewnewnew<-matrix(c(rep(1,30),X2,X3),ncol=3)
  XprimeX <- t(Xnewnewnew) %*% Xnewnewnew               
  XprimeY <- t(Xnewnewnew) %*% Y              
  invXX <- solve(XprimeX)              
  betahat <- invXX %*% XprimeY        
  L1 <- betahat
  YprimeY <- t(Y) %*% Y                 
  s2 <- (YprimeY - (t(betahat) %*% XprimeY)) / (n-p)  
  V_betahat <- (s2[1,1] * invXX)       
  SE_betahat <- (sqrt(diag(V_betahat))) 
  L2 <- SE_betahat
  
  L<-cbind(L1,L2)
  colnames(L) <- c("L[1]", "L[2]")
  print(L)
}
myAnova1intercept(Y, XF)
fit1 = lm(Y ~ XF); summary(fit1) # with an intercept

#4
n = 30; set.seed(0);
X = runif(n)*4-2;
Y = 1 + 3*X + rnorm(n);
fit1 = lm(Y ~ X); summary(fit1) # beta0_hat = 1.0161; beta1_hat = 2.9304, obtained by OLS
b0 <- summary(fit1)$coefficients[1,1]
b1 <- summary(fit1)$coefficients[2,1]
beta <- as.vector(cbind(b0,b1))
beta
sig <- sigma(fit1)
sig

#4.1
myFullObj <- function(a) {(-1)*sum(dnorm(Y, mean=a[1], sd = a[2], log=TRUE))}
myFullObj(c(beta,sig)) #sanity check at least it doesn't crash (works) 

#4.2
sigKnown = 2; myObj1 <- function(b) {myFullObj(c(b,sigKnown))};
out = optim(c(beta,sigKnown), myObj1,method = "BFGS"); out
sigKnown = 100; myObj1 <- function(b) {myFullObj(c(b,sigKnown))};
out = optim(c(beta,sigKnown), myObj1,method = "BFGS"); out
sigKnown = .25; myObj1 <- function(b) {myFullObj(c(b,sigKnown))};
out = optim(c(beta,sigKnown), myObj1,method = "BFGS"); out
beta

#4.3
out = optim(c(beta,sig), myFullObj,method="L-BFGS-B",lower=c(-10^4,10^(-4))); out 
beta
sig
            