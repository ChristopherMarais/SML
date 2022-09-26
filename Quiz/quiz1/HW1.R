#3.1
set.seed(0)
n=100
lambda=10
x=rexp(n,lambda)
x
llhd=function(lamb){
  theta=-n*log(lamb)+lamb*sum(x)
return(theta)}

lambda.mle=optimize(f=llhd,interval=c(0.1,20))$minimum
lambda.mle
#3.2
#Take the likelihood function of exponential distribution, and take the natural log of it. Then, take the first derivative with respect to lambda, and set it equal to zero. We find the MLE estimator of lambda is n/Sigma x. Take the second derivative and check if it is less than zero to make sure it is the maximum instead of a minimum. 

lambda.mle2=n/sum(x)
lambda.mle2

#4.1
n=1000;size=4
CI1=matrix(0,n,2);CI2=matrix(0,n,2);CI3=matrix(0,n,2);coverage=matrix(0,n,3)

width1=rep(0,n);width2=rep(0,n);width3=rep(0,n)

for(j in 1:n){
  set.seed(j)
  x=rnorm(size,0,1)
  CI1[j,]=mean(x)+c(-1,1)*qnorm(0.975)/sqrt(size)
  CI2[j,]=mean(x)+c(-1,1)*qt(0.975,size-1)*sd(x)/sqrt(size)
  CI3[j,]=mean(x)+c(-1,1)*qnorm(0.975)*sd(x)/sqrt(size)
  
  if(CI1[j,1]<=0 & CI1[j,2]>=0) {coverage[j,1]=1}
  if(CI2[j,1]<=0 & CI2[j,2]>=0) {coverage[j,2]=1}
  if(CI3[j,1]<=0 & CI3[j,2]>=0) {coverage[j,3]=1}
  
  width1[j]=CI1[j,2]-CI1[j,1]
  width2[j]=CI2[j,2]-CI2[j,1]
  width3[j]=CI3[j,2]-CI3[j,1]
}
freq.of.coverage=apply(coverage,2,mean)
freq.of.coverage
# The frequency of coverage is reported as 0.946, 0.958, 0.869 for the three different cases, respectively. 
#4.2

par(mfrow=c(1,2),mar=c(4,4,2,2))
hist(width2,probability = TRUE, xlab="width", ylab="propotion", main="case2")
hist(width3,probability = TRUE, xlab="width", ylab="propotion", main="case3")

#The interval width of case 1 is 1.96, since we assume the known sigma and the distribution is normal(0,1). Case 1 has the highest frequency of coverage. The histogram of interval widths of the rest 2 cases are reported. We find that in case 2, the interval width is usually wider than case 3, and it contains the true mean(0) more frequently than case 3. Case 3 has the least frequency of coverage since we used a large-sample approach even though the sample size is really small (n=4). 

