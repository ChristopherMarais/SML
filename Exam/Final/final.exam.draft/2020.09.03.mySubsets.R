# abe6933 sml/pssc

# function to generate all subsets of the set (1,2,...,p)
myf <- function(p) {
  out = matrix(c(0,1),nrow=2);
  if (p > 1) {
    for (i in (1:(p-1))) {
      d = 2^i
      o1 = cbind(rep(0,d),out)
      o2 = cbind(rep(1,d),out)
      out = rbind(o1,o2)
    }
  }
  colnames(out) <- c(2^((p-1):0)); # powers for binary expansion
  # colnames(out) <- c()
  out
}

myf(1)
myf(2)
myf(3)
o4 = myf(4)
rs = rowSums(o4)
ii = rs == 3;
o4[ii,]

nbSubsets <- function(p,m) {
  M  = myf(p)
  rs = rowSums(M)
  ii = (rs == m)
  (M[ii,])
}
nbSubsets(4,3)
nbSubsets(4,1)
nbSubsets(1,1)
nbSubsets(1,0)
nbSubsets(4,0)

# function to convert binary representation to decimal representation 
bin2dec <- function(binM) {
  dd = dim(binM);  # nrows and ncols
  p = dd[2]-1      # max power; 
  d = rep(0,dd[1]) # initialize placeholder for the answer
  for (i in 1:(p+1)) {
    d = d + 2^(p+1-i)*binM[,i]
  }
  d
}
bin2dec(o4)
a4 = cbind(o4,bin2dec(o4)); a4
