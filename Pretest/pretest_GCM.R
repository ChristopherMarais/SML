#####################
# PROBLEM 1

# 1.1
n=5
X <- seq(1, n, by = 1)

# debnsity distribution function
f_x <- function(x, n, lambda){
  numer <- (lambda^n)*(x^(n-1))*(exp(-lambda*x))
  denom <- factorial(n-1)
  Y <- numer/denom
  return(Y)
}

S_n = 0
i<-0
for (val in X){
  i<-i+1
  S_n = S_n + f_x(x=val, n=i, lambda=1)
}

S_n/n



##################
beta_est <- sum(dpois(n, 1))/n
exp_est <- dexp(x, beta_est)
exp <- rexp(x, 1)

# variance
variance <- var(exp_est)
err <- var(exp)

# bias
mse <- mean((exp_est - exp)^2)
bias <- mse - err - variance

print(variance)
print(bias)

# # 1.2
# n=5
# x <- seq(1, n, by = 1)
# beta_est <- 1/(sum(dpois(n, 1))/n)
# exp_est <- dexp(x, beta_est)
# exp <- rexp(x, 1)
# 
# # variance
# variance <- var(exp_est)
# err <- var(exp)
# 
# # bias
# mse <- mean((exp_est - exp)^2)
# bias <- mse - err - variance
# 
# print(variance)
# print(bias)



#####################
# PROBLEM 2
# 2.1
A = matrix(c(1,0,1,2,1,1,3,1,3),nrow=3,byrow=TRUE)
b = matrix(c(1,1,1),nrow=3,byrow=TRUE)
x = solve(A,b)
print(x)

# 2.2
A = matrix(c(1,0,1,2,1,1,3,1,2),nrow=3,byrow=TRUE)
solve(A,b)
solve(A)

# 2.3 
det(A)
eigen_v <- eigen(A)
solution <- eigen_v$vectors[,3]
#####################
# PROBLEM 3
# 3.2 
crit_val = qt(0.975,22)
t_val = (1-1.6882)/0.7044
comp_vals = crit_val-t_val
p_val = 2*pt(t_val,22)


# 3.5
3 - 1.2717 + 1.6882*1 -1.3406*1

#####################
# PROBLEM 4
# Convert binary string to decimal value
bin_to_dec <- function(vec) {
  dec = 0
  len <- length(vec)
  i<-len
  for (val in vec){
    i<- i-1
    dec <- dec + (val*(2^i))
  }
  return(dec)
}
# Convert decimal to binary value
dec_to_bin <- function(x, p) {
  i <- 0
  string <- c(numeric(p))
  while(x > 0) {
    string[p - i] <- x %% 2
    x <- x %/% 2
    i <- i + 1
  }
  return(string)
}

# get subsets function
getSubsets <- function(p,m) {
  max_num <- (2^p)-1
  all_nums <- seq(max_num)
  mat <- matrix(data=NA, nrow=max_num, ncol=p)
  i<-1
  for (j in all_nums){
    mat[i,] <- dec_to_bin(j, p)
    i<-i+1
  }
  mat[rowSums(mat)==m,]
}

bin2dec <- function(mat){
  num_vec <- c()
  if(nrow(mat)==0){
    num_vec <- numeric(ncol(mat))
  } else {
    for (l in seq(nrow(mat))){
      x <- bin_to_dec(mat[l,])
      num_vec <- append(num_vec, x)
    }
  }
  return(num_vec)
}

bin2dec(getSubsets(3,0))


# I ran out of time to implement Bill Gosper's way of doing this.

#####################
# PROBLEM 5
