import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

df2=pd.read_csv('C:\\Users\\alex314\\Desktop\\LoadProject\\nyiso.csv')
load_NYC=df2[['Date','N_Y_C_']]
load_NYC.Date=pd.to_datetime(load_NYC.Date[:45000], format='%d-%b-%Y %H:%M:%S')
load_NYC.columns=['Date','Load']
load_NYC=load_NYC.dropna()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot_date(load_NYC.Date[:45000], load_NYC.Load[:45000], 'b-',tz=None, xdate=True, ydate=False)
ax.set_title('Electric Power Load for New York City Zone')
ax.set_ylabel('Electric Load, MW')

data.replace([-999, -1000], np.nan)


tolerance = 100
idxbad = (abs(y(idx)-yy) > tolerance)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plt.plot(xs,ys)
ax1.set_title('Actual and Smoothed Loads') 
         
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot_date(dates, df_cleaned.Load[-10000:]-lm.predict(Xtest), 'r-',tz=None, xdate=True,
          ydate=False)
ax1.set_title('Difference of Actual and Predicted Loads, MW')
plt.show()