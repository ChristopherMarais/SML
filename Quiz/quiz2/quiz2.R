y = c(0.75, 0.66, -0.18, -0.53, 2.12)
set.seed(0)
x = runif(5)*2 - 1
x

df = data.frame(x, y)
linear_reg = lm(y ~ 0 + ., df)
plot(linear_reg)
linear_reg
