import pandas as pd
from os import listdir

path='C:\\Users\\alex314\\Desktop\\LoadProject\\Weather\\WeatherHourly'

stations = pd.Series(['KALB', 'KART', 'KBGM', 'KBUF', 'KELM', 'KHPN', 'KISP', 'KJFK', 'KLGA',
     'KMSS', 'KMSV', 'KPBG', 'KPOU', 'KROC', 'KSWF', 'KSYR', 'KRME'])

WBANs=[14735, 94790, 4725, 14733, 14748, 94745, 4781, 94789, 14732, 94761, 
        54746, 64776, 14757, 14768, 14714, 14771, 64775]

stationTable=dict(zip(WBANs, stations))
        
#Aggregating weather files
result=[]
for file in listdir(path):
    df=pd.read_csv(path+'\\'+file,usecols=[
        'WBAN', 'Date','Time','DryBulbFarenheit', 'DewPointFarenheit'], low_memory=False)
    df=df[df.WBAN.isin(WBANs)] #Select a subset of WBAN's
    result.append(df)

weather=pd.concat(result)
weather.columns=['Name','Date','Time', 'Temperature','DewPoint']
weather=weather[weather.Temperature != 'M']
weather=weather[weather.DewPoint != 'M']
weather=weather.replace({"Name": stationTable})
weather.Temperature.apply(int)
weather.DewPoint.apply(int)
weather=weather.pivot(index='Date', columns='Name', values='Temperature')
weather.to_csv('C:\\Users\\alex314\\Desktop\\LoadProject\\Load1\\weather_data.csv')






