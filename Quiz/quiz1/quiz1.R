#####
# Q1
# function
myPdf <- function(x){
  fv = dt(x+1,2)/4+dt(x-1,2)/4+dnorm(x)/2
  ii=(abs(x)>3)
  fv[ii]=0
  fv/(0.9384766)
}


# plot density function
set.seed(42)
n = 100
X = seq(-3,3,length.out=n)
p = myPdf(X)
plot(X, p)

# calculate variance
first <- function(x){
  return(x*myPdf(x))
}
second <- function(x){
  return((x^2)*myPdf(x))
}
m1 = integrate(first, -3, 3)[[1]]
m2 = integrate(second, -3, 3)[[1]]
variance = m2-m1**2
variance
#####
#Q2
integrate(myPdf, 1, 3)

#####
#Q3
#f(x,y) = f(x)f(y) # then two variables are independent
#f(x) = integral(f(x,y))*dy
#f(y) = integral(f(x,y))*dx


#####
#Q4
# False 


#####
#Q5
cdf_t <- function(x){
  return(1-(1/(x^2)))
}

# pdf is the derivative of cdf
pdf_5=function(x){
  2*x^(-3)
}
p_5=function(x){
  x*c(x)
}
mean=integrate(p_5,1,Inf)[[1]]
y=mean*2-1
print(y)
