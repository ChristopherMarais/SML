###########
#1
set.seed(0)
y = c(0.75, 0.66, -0.18, -0.53, 2.12)
x = runif(5)*2 - 1
df = data.frame(x, y)

obj_fn <- function(data, par) {
  return(sum((df['y']-(df['x']*par[1]))^2))}

b = optim(par = c(0.1), 
      fn=obj_fn, 
      data=df, 
      method = "Brent", 
      lower = -10.0, 
      upper = 10.0)



###########
#2
#The degree 3 polynomial will have the lower 
# residual sum (RRS) of squares on the training data. 
# This is because the higher order polynomial is more flexible
# and can allow curves that will reach closer to more points of
# the data points

###########
#3
# No, This does not indicate a causal relationship, but only taht the 
# there is a linear relationship between the features

###########
#4
petType = rep(c("C","D"),2)

func <- function(data){
  num_vec = ifelse(data == "C", 1, 0)
  return(num_vec)
}

###########
#5
# False R is the sample correlation coefficient and R squared is 
# The square of this value. 



