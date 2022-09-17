# WH1
#####
#1
# import data and break up into train/validation/test sets
data_df <- read.csv("SML.NN.data.csv")
train_df = data_df[data_df$set == 'train',]
valid_df = data_df[data_df$set == 'valid',]
train_val_df = data_df[data_df$set != 'test',]
test_df = data_df[data_df$set == 'test',]

#1.1
getClass1Prop <- function(x, r, data=train_val_df) {
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

getClass1Prop(x=c(5,9), r=10, data=valid_df)

