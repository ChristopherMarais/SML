########################################
# HOMEWORK 1 2022-09-16
########################################
##########
#1
# import data and break up into train/validation/test sets
data_df <- read.csv("SML.NN.data.csv")
train_df = data_df[data_df$set == 'train',]
valid_df = data_df[data_df$set == 'valid',]
test_df = data_df[data_df$set == 'test',]

#####
#1.1
# function to get proportion of class 1 in radius to point x (single)
# output is a float
getClass1Prop <- function(x, r, data=train_df) {
  "Import the data this function is 
  based on before using it. The data should be named data_df
  and contain columns Y, X1, X2, set all in a dataframe.
  x is a vector of length 2
  r is the selected radius
  "
  # calculate the distance between the point and all the data
  x1 = (data$X1-as.numeric(x[1]))^2
  x2 = (data$X2-as.numeric(x[2]))^2
  data$euc_dist = sqrt(rowSums(data.frame(x1,x2)))
  #select points inside radius
  in_r_df = data[data$euc_dist <= r,] 
  if(nrow(in_r_df)==0){
    # return NA if no points within radius
    return(NA) 
  } else {
    # calculate proportion of class 1 values in data
    class_1_prop = sum(in_r_df$Y)/nrow(data)
    return(class_1_prop)
  }
}

#####
#1.2
# get class 1 prediction for points in x (multiple)
# output is a vector of binary predictions
getClass1Prediction <- function(x, r, t=0.5, data=train_df){
  "t is the threshold and should be between 0 and 1."
  pred_vec = c()
  # loop through all points in x
  for(i in 1:nrow(x)){
    x_i = x[c('X1','X2')][i,] # get coordinates
    class_1_prop = getClass1Prop(x=x_i, r=r, data=data)
    if(class_1_prop>=t){
      pred=1
    }else{
      pred=0
    }
    pred_vec=c(pred_vec, pred)
  }
  return(pred_vec)
}

# function to get confusion matrix of binary prediction
# output is a data frame in format:
# (true positive, false negative)
# (false positive, true negative)
getConfusionMatrix <-function(true, pred){
  "true is the real class of the data
  pred is the predicted class of the same data"
  df = data.frame(true, pred)
  # calculate equality of data
  df$equal = df$true == df$pred
  # calculate tp, tn, fp, and fn
  tp = as.numeric(nrow(df[(df$equal == TRUE) & (df$pred == 1),]))
  tn = as.numeric(nrow(df[(df$equal == TRUE) & (df$pred == 0),]))
  fp = as.numeric(nrow(df[(df$equal == FALSE) & (df$pred == 1),]))
  fn = as.numeric(nrow(df[(df$equal == FALSE) & (df$pred == 0),]))
  # save values in confusion matrix data frame
  conf_mat_df = setNames(
    data.frame(
      c(tp, fp), 
      c(fn, tn),
      row.names = c('1','0')),
    c('1','0'))
  return(conf_mat_df)
}

# function to get the 
getMissClassRate <-function(true, pred){
  " Input is true vector and prediction vector
  output is a ratio of wrong classifications.
  "
  conf_mat_df = getConfusionMatrix(true, pred)
  miss_class_rate = (conf_mat_df['1','0'] + conf_mat_df['0','1'])/sum(conf_mat_df)
  return(miss_class_rate)
}

#####
#1.3
# make plot of coordinates density distribution
plot(train_df[train_df$Y==1,]$X1, 
     train_df[train_df$Y==1,]$X2, 
     main = "Distribution of classes",
     xlab = "X1", 
     ylab = "X2",
     pch = 15, 
     col = "blue")
points(train_df[train_df$Y==0,]$X1, train_df[train_df$Y==0,]$X2, pch = 0, col = "blue")
points(valid_df[valid_df$Y==1,]$X1, valid_df[valid_df$Y==1,]$X2, pch = 19, col = "red")
points(valid_df[valid_df$Y==0,]$X1, valid_df[valid_df$Y==0,]$X2, pch = 1, col = "red")

# 0.5 seems like a good number for r

#####
#1.4
# select best r for range of r values (grid search)

getMissClassRate(true = valid_df$Y,
                   pred = getClass1Prediction(x=valid_df, r=0.5, t=0.5, data=train_df))

#####
#1.5
# Already reduced loops to only one

#####
#1.6
# KNN
# get lowest missclassrate for KNN and compare to lowest RNN