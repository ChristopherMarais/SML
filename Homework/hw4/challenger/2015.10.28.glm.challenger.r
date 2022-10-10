# Challenger o-rings logistic regression example

# change mydir to the one where the challenger.dat is contained
setwd(mydir)

ch0 <- read.fwf("challenger.dat", width=c(8,8,8,8), col.names=c("flight", "degrees", "numfail", "indfail"))

attach(ch0)

chmod1 <- glm(formula=indfail ~ degrees, family=binomial("logit"))
summary(chmod1)

anova(chmod1, test="Chisq")

