import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from word2number import w2n
 

df = pd.read_csv('carprices.csv')

dum = pd.get_dummies(df.Car_Model,dtype=int)
mer = pd.concat([df,dum],axis='columns')
fin = mer.drop(['Car_Model','BMW X5'],axis='columns')
model = LinearRegression()
x = fin.drop('Sell_Price',axis='columns')
y = fin.Sell_Price
x1 = x.values
model.fit(x1,y)
k=model.predict([[45000,4,0,1]]) 
print(k)
