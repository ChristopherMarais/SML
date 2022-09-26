
##################
# ISLR Chapter 3
##################
#############
# Question 4
#############
#####
# a.)
#####

"4.
(a) The extra polynomial terms allow for a closer fit 
(more degrees of freedom) of the training data, so I would 
expect the training RSS for cubic regression to be lower than 
for simple linear regression.

(b) The true relationship is linear and so simple linear regression
would generalize better to unseen data, as such I would expect it 
to have lower test RSS. The cubic model likely over fit the training 
data, and so I would expect it to have a higher test RSS.

(c) Cubic regression will have a better fit to non-linear data and so 
its training RSS will be lower.

(d) The test RSS depends on how far from linear the true relationship$f(x)$ is.
If $f(x)$ is more linear than cubic, then cubic regression can over fit,
so cubic RSS will be higher and liner RSS will be lower.
If $f(x)$ is more cubic than linear, then linear regression can under fit, 
so linear RSS will be higher and cubic RSS will be lower."



#############
# Question 10
#############
#####
# a.)
#####
library("ISLR2")
carseats_lm = lm(Sales~Price+Urban+US,data=Carseats)
summary(carseats_lm)

#####
# b.)
#####

"
- The intercept represents the number of car seats sold on average when all other predictors are disregarded.
- The `Price` coefficient is negative and so sales will fall by roughly 54 seats(0.054x1000)for every unit($1) increase in price.
- The `Urban=Yes` coeff is not statistically significant. The `US=Yes` coeff is 1.2, and this means an average increase in car seat sales of 1200 units when `US=Yes`(this predictor likely refers to the shop being in the USA).

"

#####
# c.)
#####
attach(Carseats)
contrasts(US)
contrasts(Urban)

#$$Sales = 13.04\ + -0.05Price \ + -0.02Urban(Yes:1,No:0) \ + 1.20US(Yes:1,No:0)$$

#####
# d.)
#####
carseats_all_lm = lm(Sales~.,data=Carseats)
summary(carseats_all_lm)
#- Null hypothesis can be rejected for `CompPrice`, `Income`, `Advertising`, `Price`, `ShelvelocGood`, `ShelvelocMedium` and `Age`.

#####
# e.)
#####
carseats_all_lm2 = lm(Sales~.-Education-Urban-US-Population,data=Carseats)
summary(carseats_all_lm2)

#####
# f.)
#####
#- The RSE goes down from 2.47 __model (a)__ to 1.02 __model (e)__. The R2 statistic goes up from 0.24 __(a)__ to 0.872 __(e)__ and the F-statistic goes up from 41.52 to 381.4. 
#- The statistical evidence clearly shows that __(e)__ is a much better fit.


#####
# g.)
#####
confint(carseats_all_lm2)

#############
# Problem 1
#############
#####
# a.)
#####