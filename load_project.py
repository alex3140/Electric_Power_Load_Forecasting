#Script 1: Model Development
#Builds forecasting models based on linear regression, tree bagging,
#and gradient boosting

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor

merged_data=pd.read_csv(
            'C:\\Users\\alex314\\Desktop\\LoadProject\\merged_data.csv')

#Select the data corresponding to the NYC zone
modeldata=merged_data[['Date','N_Y_C_','TemperatureKLGA_1','TemperatureKLGA_2']]
modeldata.columns=['Date','Load','Temperature','DewPoint']

#Create temporal predictors
modeldata['Date']=pd.to_datetime(modeldata.Date, format='%d-%b-%Y %H:%M:%S')

modeldata['Hour']=pd.Series(
                    [modeldata.Date[idx].hour for idx in modeldata.index])
modeldata['Year']=pd.Series(
                    [modeldata.Date[idx].year for idx in modeldata.index])
modeldata['Month']=pd.Series(
                    [modeldata.Date[idx].month for idx in modeldata.index])

#There are important differences in load between weekdays and weekends
#Create suitable predictors:
modeldata['DayOfWeek']=pd.Series(
    [modeldata.Date[idx].isoweekday() for idx in modeldata.index])
modeldata['isWeekend']=pd.Series(
 [int(modeldata.Date[idx].isoweekday() in [1,7]) for idx in modeldata.index])

#Lagged predictors:
modeldata['PriorDay']=modeldata.Load.shift(24)
modeldata['PriorWeek']=modeldata.Load.shift(168)

modeldata=modeldata.dropna()
        
#Plot electric power load vs. time
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot_date(modeldata.Date[:45000], modeldata.Load[:45000],\
            'b-',tz=None, xdate=True, ydate=False)
ax.set_title('Electric Power Load for New York City Zone')
ax.set_ylabel('Electric Load, MW')

features=['Hour', 'Month','DayOfWeek','Temperature', \
            'isWeekend','DewPoint','PriorDay','PriorWeek']
X=modeldata[features]
Y = modeldata.Load

#Scatter plots
g=sns.PairGrid(modeldata,vars=['Load', 'Temperature', 'PriorDay'])
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=20)

font = {'family' : 'normal', 'size'   : 20}
matplotlib.rc('font', **font)
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(X.Month, Y)
ax1.set_xlabel("Month")
ax1.set_ylabel("Load, MW")
ax1.set_xlim([.5, 12.5])
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(X.Temperature, Y)
ax3.set_ylabel("Load, MW")
ax3.set_xlabel("Air Temperature, F")
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(X.PriorDay, Y)
ax4.set_ylabel("Load, MW")
ax4.set_xlabel("Prior Day Load, MW")
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(X.Hour, Y)
ax2.set_xlim([-1, 25])
ax2.set_xlabel("Hour")
ax2.set_ylabel("Load, MW")

#Split the data into testing and training sets
Xtrain, Xtest=X[:45000], X[45000:50000]
Ytrain, Ytest=Y[:45000], Y[45000:50000]
DateTest=modeldata.Date[45000:50000]

##############################################
#Linear regression model
lm = linear_model.LinearRegression()
lm.fit(Xtrain, Ytrain)

print('Coefficients: \n', lm.coef_)
print("Residual sum of squares: %.2f"%np.mean((lm.predict(Xtest) - Ytest) ** 2))
print('Variance score: %.2f' % lm.score(Xtest, Ytest))

#MSE on testing and training sets
MSEs_lm=[mean_squared_error(
    Ytest, lm.predict(Xtest)), mean_squared_error(Ytrain, lm.predict(Xtrain))]

print 'MSE on the training set:',mean_squared_error(Ytrain, lm.predict(Xtrain))
print 'MSE on the testing set:', mean_squared_error(Ytest, lm.predict(Xtest))

#Actual and Predicted Load
dates=pd.to_datetime(modeldata.Date[45000:50000], format='%d-%b-%Y %H:%M:%S')
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot_date(dates, modeldata.Load[45000:50000], 'r-', xdate=True,
          ydate=False, label='Actual')
ax1.set_title('Actual and Predicted Loads')          
ax1.plot(dates, lm.predict(Xtest), 'g-', label='Predicted')
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot_date(dates, modeldata.Load[45000:50000]-lm.predict(Xtest), 'r-',
                tz=None, xdate=True, ydate=False)
ax1.set_ylabel('Load, MW')
ax2.set_ylabel('Error, MW')
ax2.set_title('Error between actual and predicted load')
ax1.legend()

###################################
#Gradient Boosting Model
#Default parameters:
#n_estimators=100- number of boosting stages to perform
#learning_rate=.1 shrinks the contribution of each tree by learning_rate
#max_depth=3-maximum depth of the individual regression estimators
gradBoost=GradientBoostingRegressor()
gradBoost.fit(Xtrain, Ytrain)

#MSE on testing and training sets
MSEs_Boost=[mean_squared_error(Ytest, gradBoost.predict(Xtest)), \
            mean_squared_error(Ytrain, gradBoost.predict(Xtrain))]

print("MSE on test data:", mean_squared_error(Ytest, gradBoost.predict(Xtest))) 
print("MSE on training data:", 
        mean_squared_error(Ytrain, gradBoost.predict(Xtrain)))

#Actual and Predicted Load
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot_date(dates, modeldata.Load[45000:50000], 'r-',tz=None, xdate=True,
          ydate=False, label='Actual Load')
ax1.set_title('Gradient Boosting: Actual and Predicted Loads')          
plt.plot(dates, gradBoost.predict(Xtest), 'g-',label='Predicted Load')
ax1.legend()
ax1.set_ylabel("Load, MW")
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot_date(dates, modeldata.Load[45000:50000]-gradBoost.predict(Xtest), 
'r-',tz=None, xdate=True, ydate=False)
ax2.set_title('Error between actual and predicted loads')
ax2.set_ylabel("Error, MW")

featImportances=gradBoost.feature_importances_
pos= arange(len(features))+0.7
fig, ax = plt.subplots()
plt.yticks(pos,features)
plt.barh(pos,featImportances, 1, color="blue")
ax.set_title('Gradient Boosting: Feature Importance')

##############################################################################
#Tree Bagging
TreeBagger=BaggingRegressor()
TreeBagger.fit(Xtrain, Ytrain)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot_date(dates, modeldata.Load[45000:50000], 'r-',tz=None, xdate=True,
          ydate=False, label='Actual Load')
ax1.set_title('Tree Bagging: Actual and Predicted Loads')          
plt.plot(dates, TreeBagger.predict(Xtest), 'g-',label='Predicted Load')
ax1.legend()
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot_date(dates, modeldata.Load[45000:50000]-TreeBagger.predict(Xtest), 'r-',tz=None, xdate=True,
          ydate=False)
ax2.set_title('Error between actual and predicted loads, MW')

MSEs_Bagging=[mean_squared_error(Ytest, TreeBagger.predict(Xtest)), mean_squared_error(Ytrain, TreeBagger.predict(Xtrain))]

########################################
#Model Comparison: Bar charts
fig, ax = plt.subplots()
width=.3
rects1 = ax.bar([0,1,2], [MSEs_Boost[0],MSEs_lm[0], MSEs_Bagging[0]], width, color='y')
rects2 = ax.bar([width, width+1, width+2], [MSEs_Boost[1],MSEs_lm[1], MSEs_Bagging[1]], width, color='b')
ax.set_xticks([width, width+1, width+2])
ax.set_xticklabels(('Gradient Boosting', 'Linear Model', 'Tree Bagging'))
ax.set_title('Comparison of models on training and testing sets')
ax.set_ylabel('Mean Squared Error')
ax.legend((rects1[0], rects2[0]), ('Testing Set', 'Training Set'))
#Gradient boosting is fairly robust to overfitting 