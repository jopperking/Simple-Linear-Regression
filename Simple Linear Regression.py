import matplotlib.pyplot as plt
import pandas as pd
# import pylab as pl
import numpy as np
# %matplotlib inline


# in terminal type : wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv
df = pd.read_csv("FuelConsumption.csv")

print(df.head(10))

# summarize the data
print(df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(10)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]

viz.boxplot()
plt.show()

viz.hist() #histogram
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel("CO2EMISSIONS")
plt.show()

## `Splite Data to 80% train & 20% test`

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
print (msk)
print (~msk)
print (train)
print (test)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='red')
plt.scatter(test.ENGINESIZE,test.CO2EMISSIONS,color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel("CO2EMISSIONS")
plt.show()

## **Training**

from sklearn  import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x,train_y)
#The coefficients
teta0 = regr.intercept_
teta1 = regr.coef_
print ('Coefficients: ',teta1)
print ('Intercept: ' , teta0)

## Regression result formula

y = teta1[0][0] *  train_x + teta0[0]

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,y,'-r')
plt.show()

## Error Evaluation

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(np.asanyarray(test["CO2EMISSIONS"])) # real values
test_y_ = regr.predict(test_x)  # Guessed values via regression

print ("Mean absolute error : %.2f" %np.mean(np.absolute(test_y_ - test_y))) # absolute = قدر مطلق
print ("Residual sum of squares (MSE) : %.2f" %np.mean(np.absolute(test_y_ - test_y) ** 2))
print ("R2-score: %0.2f" %r2_score(test_y , test_y_) )


## Test

print ('Test ENGINESIZE : 2.0 :' )
print ('Guessed value of CO2EMISSIONS : ' , regr.predict(np.array([[2.0]])) )
