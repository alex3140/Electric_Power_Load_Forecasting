import urllib
import pandas as pd
from datetime import datetime
 
dates=pd.date_range(start='1/1/2007', periods=20,freq='MS').date
path='C:\\Users\\alex314\\Desktop\\LoadProject\\Load1'

count=0
for date in dates:
    count+=1
    print "Downloading file %d out of %d \n" %(count, len(dates))
    name=datetime.strftime(date, '%Y%m%d')+'pal_csv.zip'
    url='http://mis.nyiso.com/public/csv/pal/'+name
    path_to_file = path + '\\'+name
    file = urllib.URLopener()
    file.retrieve(url, path_to_file)

