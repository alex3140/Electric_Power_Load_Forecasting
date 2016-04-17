import urllib
import pandas as pd
 
dates=pd.date_range(start='5/1/2007', periods=50,freq='MS').date
path='C:\\Users\\alex314\\Desktop\\LoadProject\\Weather'

count=0
for date in dates:
    count+=1
    print "Downloading file %d out of %d \n" %(count, len(dates))
    name='QCLCD'+datetime.strftime(date, '%Y%m')+'.zip'
    url='https://www.ncdc.noaa.gov/orders/qclcd/'+name
    path_to_file = path + '\\'+name
    file = urllib.URLopener()
    file.retrieve(url, path_to_file)

