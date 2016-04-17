from pandasql import *
import pandas as pd

pysqldf = lambda q: sqldf(q, globals())

load=pd.read_csv('C:\\Users\\alex314\\Desktop\\LoadProject\\load_data.csv').ix[:20000]
weather=pd.read_csv('C:\\Users\\alex314\\Desktop\\LoadProject\\weather_data.csv').ix[:20000]

q  = """
SELECT *
FROM load
INNER JOIN weather
ON load.Date = weather.Date;
"""

df = pysqldf(q)
df.to_csv('C:\\Users\\alex314\\Desktop\\LoadProject\\merged_data1.csv')
