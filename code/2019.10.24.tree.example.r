
# Good and bad (single) trees:

# install.packages("tree"); 
library(tree)

set.seed(0); n = 1000; # you can try larger values of n as well
M = matrix(runif(n*2),ncol=2)
X = 2*M - 1; # features are uniform on [-1,1]x[-1,1]
X = round(X*1000)/1000; # round to 3 decimal places for ease of tree display
ii = X[,1]*X[,2]>0 # indices of (x1,x2) in Q1 or Q3.
#
# "Ugly" example: the first split (bad) leads to trees more complex than truth;
cmu = 1; s = 0.1; 
plot(X[,1],X[,2],type="p")
points(X[ii,1],X[ii,2],type="p",col="red")
Y = rep(cmu,n); Y[ii] = -cmu; # Q1 and Q3 mean is -cmu; Q2 and Q4 mean is cmu
Y = Y + rnorm(n,sd=s)
t1 = tree(Y ~ X); t1
summary(t1)
t2 = tree(Y ~ X,control=tree.control(nobs=n,mindev=0.001)); t2 # also try mindev=0.005
summary(t2)
plot(t2,type="uniform"); text(t2,digits=2)

# "Good" example: true partition is recovered accurately
cmu = 1; s = 0.1; 
i1 = X[,1] > 0; i2 = X[,2] > 0;
Y = rep(cmu,n); # quadrant 1 (Q1) mean is 1*cmu
Y[i1 & (!i2)]  = 2*cmu; # Q2 mean is 2*cmu
Y[(!i1)&(!i2)] = 3*cmu; # Q3 mean is 3*cmu
Y[(!i1)&(i2)]  = 4*cmu; # Q4 mean is 4*cmu
Y = Y + rnorm(n,sd=s); # add noise to the mean vector
t1 = tree(Y ~ X); t1
summary(t1)
plot(t1, type="uniform"); text(t1, digits=2)
