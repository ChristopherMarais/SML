# effect of multiple testing on feature prescreening

# for a problem with nobs = n, generate p predictors as random N(0,1) noise ...
#       and compute their correlations with the vector of responses;
#       assumes y = c(rep(0,n/2),rep(1,n/2));
myF <- function(n,p,seed=0) {
  set.seed(seed)
  y = c(rep(0,n/2),rep(1,n/2))
  M = matrix(rnorm(n*p),nrow=n)
  outV = rep(0,p)
  for (i in 1:p) {
    outV[i] = cor(y,M[,i])
  }
  outV
}
o1 = myF(10,1);  o1
o1 = myF(10,10); max(abs(o1))
o1 = myF(100,1); o1
o1 = myF(100,10); max(abs(o1))

# see how max correlation grows when n=10 and p = 10*2^j, j=0,1,...
for (j in 0:10) { print(max(myF(10, 10*2^j))) }

# see how max correlation grows when n=50 and p = 10*2^j, j=0,1,...
for (j in 0:10) { print(max(myF(50, 10*2^j))) }

# ... can also look at max abs correlation;

outV = myF(50, 10*2^j);
jj = which(abs(outV) == max(abs(outV))) # determines which covariate is most correlated
outV[jj]

## effect of such prescreening on MER on the entire dataset
n = 50; p = 10*2^10;
set.seed(0)
y = c(rep(0,n/2),rep(1,n/2))
M = matrix(rnorm(n*p),nrow=n)
g1 = glm(y ~ M[,jj], family=binomial) # univariate logistic reg with the "best" covariate
summary(g1)
fv = g1$fitted.values
mean(abs(y-(fv>0.5))) # whole data MER
#
## train/test split;
set.seed(100)
test.prop = 0.3;
test.id = sample(n); test.id = test.id[1:(floor(test.prop*n))]
train.id = setdiff(1:n,test.id)
my.df = data.frame(y = y, x = M[,jj])
g2 = glm(y ~ x, data=my.df, subset=train.id, family=binomial);
summary(g2)
pv = predict.glm(g2,my.df[test.id,]); # test 
mean(abs(y[test.id]-(pv>0.5))) # test data MER


