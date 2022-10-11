#####1 
# the OLS problem has multiple solutions - infinitely many
set.seed(1)
p = 3
n = p*2
X = matrix(1,nrow=n,ncol=p)
X[,2] = c(3,4,5,6,7,8)
X[,3] = c(4,5,6,7,8,9)
y = runif(n,0,1)
reg=lm(y~X[,1]+X[,2]+X[,3])
summary(reg)
#we see that there is no solution for X1 and X3

#####2a 
X[,2] = runif(n,0,1)
X[,3] = runif(n,0,1)
reg=lm(y~X[,1]+X[,2]+X[,3])
summary(reg)
# False, the correlations are clearly not zero.
# just because they are not linearly dependent does
# not mean these features are not correlated

#####2b
# False, linear dependence does not give a perfect 1 or -1 correlation. 

#####3
summary(reg)
fit_vals = reg$fitted.values
cor(y, fit_vals)**2
# True, Multiple R-squared in the summary is the same as the 
# squared correlation between the two.

#####4
# The rank of the matrix is the same as
# the number of linearly independent columns.
# This can be found by firstly calculating the R-squared
# for each  pairwise combination of columns in X.
# Next, remove any columns from X that have an
# R-squared of 1 (or that is too high).
# Repeat this process until no columns are
# left or until no columns have an R-squared of 1. The number
# of remaining columns is equal to the rank of the matrix X.
 
#####5
# True