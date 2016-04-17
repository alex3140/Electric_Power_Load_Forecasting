import pandas as pd
from os import listdir

path='C:\\Users\\alex314\\Desktop\\LoadProject\\Load2'

#Aggregating load files
result=[]
for file in listdir(path):
    print "Reading file"+file+"\n"
    df=pd.read_csv(path+'\\'+file)
    df.Date=pd.to_datetime(df['Time Stamp'], format='%m/%d/%Y %H:%M:%S')
    df['Hour']=pd.Series([df.Date[idx].hour for idx in df.index])
    df.drop_duplicates(['Hour','Name'])
    df=df[['Time Stamp', 'Name', 'Load']]
    result.append(df)

result=pd.concat(result)

#Reformatting and reshaping
result.columns=['Date','Name','Load']
result=result.pivot(index='Date', columns='Name', values='Load')
result.to_csv('C:\\Users\\alex314\\Desktop\\LoadProject\\Load1\\load_data.csv')

