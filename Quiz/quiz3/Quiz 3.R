#Q1
y = c(0.75, 0.66, -0.18, -0.53, 2.12)
x1=c(1,1,1,1,1)
x2=c(3,2,3,4,2)
x3=c(4,3,4,5,3) #X3 is a linear combination of x1 and x2
reg=lm(y~x1+x2+x3)
summary(reg)
#We can see from the regression results that there is no solution to this system if x3 is a linear combination of x1 and x2.

#Q2.a
#False. Linear dependency does not necessary imply uncorrelation. For example, x1 and x2 are linearly independent in a bivariate normal distribution. They can still be correlated to each other, depending on the value of Rho. 

#Q3
#True.
n = 30; set.seed(0); p = 3;
X = matrix(runif(n*p),nrow=n)*2-1;
b = seq(1,p,by=1);
y = X%*%b + rnorm(n);

fit1 = lm(y ~ X); summary(fit1); # regression with an intercept
yhat=predict(fit1)

a=sum((y-mean(y))*(yhat-mean(yhat)))
b=sum((y-mean(y))^2)
c=sum((yhat-mean(yhat))^2)
d=(a*b)^0.5
(a/d)^2 
#This is equal to R^2 in the regression. 

#Q4
#First include all variables in myRsq. Then use backward induction-consider dropping a column of x, see if the R squared is influenced.If R squared does not change, then this column is redundant.Repeat the process until all redundant columns are eliminated, then the rank is equal to the number of remaining columns. 

#Q5
#True. 
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
#assuming e~normal e=Y-b0-X*b1
myFullObj <- function(a) {(-1)*sum(dnorm(Y-a[1]-(X*a[2]), mean=0, sd = a[3], log=TRUE))}
myFullObj(c(b0,b1,sig)) #sanity check at least it doesn't crash (works) 
#optimization starting at b0,b1
sigKnown = 2; myObj1 <- function(b) {(-1)*sum(dnorm(Y-b[1]-(X*b[2]), mean=0, sd = sigKnown, log=TRUE))}
out = optim(c(b0,b1), myObj1,method = "BFGS"); out

#we can see that with the current assumptions, the estimated beta from OLS and MLE are the same. 