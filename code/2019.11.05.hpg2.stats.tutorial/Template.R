rm(list=ls()) #clear anything in memory
setwd("/ufrc/bliznyuk/hmerrill/StatLearning") #set wd

n = 100 #set sample size
set.seed(AA) #set random seed- important to be different for each job!

x = rnorm(n) #generate n normal random variables
results_AA = mean(x) #calculate the mean

M = "20160919_results.csv" #name of file
does.M.exist = file.exists(M) #does the file already exist?
if (!does.M.exist) file.create(M) #if not, create it
write.table(results_AA, #write out results... 
            file = M, #...into the file...
            sep=",", #...comma separated...
            row.names=FALSE, #...without row names...
            append=TRUE, #...appending these lines under the existing ones...
            col.names=!does.M.exist) #...and only output column names the first time.