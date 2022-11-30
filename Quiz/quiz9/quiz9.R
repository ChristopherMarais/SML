# calculate covariance matrix
cov_mat <- (t(X) %*% X) / (nrow(X) - 1)

# get eigen values or eigen vectors
cov_e <- eigen(cov_m)


X = U.D.W^T
C=X^⊤X/(n−1)

C = WDU^T UDW^T/(n-1)

C = W D/(n-1) W^T

thus the principal components are 

XW = UDW^TW = UD

U is the unitary matrix
D is the diagonal matrix 
W is the matrix of eigen vectors
