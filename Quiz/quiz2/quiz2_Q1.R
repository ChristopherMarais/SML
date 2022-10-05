#1
set.seed(0)
y = c(0.75, 0.66, -0.18, -0.53, 2.12)
x = runif(5)*2 - 1
df = data.frame(x, y)

obj_fn <- function(data, par) {
  return(sum((df['y']-(df['x']*par[1]))^2))}

optim(par = c(0.1), 
      fn=obj_fn, 
      data=df, 
      method = "Brent", 
      lower = -10.0, 
      upper = 10.0)