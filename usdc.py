import pandas as pd
import numpy as np
import yfinance
import matplotlib.pyplot as plt
import matplotlib.dates
import datetime as dt
import pandas_datareader as web
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import os
import tkinter
from tkinter import Button,PhotoImage,Image
from tkinter import Label
from sklearn.metrics import mean_absolute_percentage_error
window = tkinter.Tk()
#window.attributes('-fullscreen', True)
window.title("USDC-USD PRICE PREDICTOR")   
cwd=os.getcwd()

crypto_currency='USDC'
against_currency='USD'
start=dt.datetime(2021,1,1)
end=dt.datetime(2022,4,17)
data= web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)


print(data.head())
print(data.tail())

X = data['Close']
X = np.array(X).reshape(-1,1)


x_data = []
y_data = []
column_len = 25
for i in range(len(X) - column_len+1):
	x_data.append(X[i:i+column_len-1,0])
	y_data.append(X[i+column_len-1,0])
	    
x_data = np.array(x_data)
y_data = np.array(y_data)

model = RandomForestRegressor(n_estimators=200)
model.fit(x_data, y_data)

y_test = y_data[-100:]
x_test = x_data[-100:]
print("\nModel Score is\n",model.score(x_test,y_test.reshape(-1,1)))
	
c = X[len(X)-column_len+1:]
a = 30
j=0
y_true=[0.999873,
0.999897,
0.999847,
1.000282,
1.000199,
1.000004,
0.999963,
1.000218,
1.000127,
1.000321,
1.000029,
1.000023,
1.000295,
1.000381,
1.00006,
1.000119,
1.000077,
0.999958,
1.000072,
0.999906,
1.000087,
0.999595,
0.999828,
1.000766,
1.000659,
1.000347,
1.000381,
1.000183,
1.000581,
1.000101
]	
y_true=np.array(y_true)
y_pred=[]
absolute_error=[]
sum_error=0
for j in range(a):
	x = model.predict(c.reshape(1,-1)).reshape(-1,1)
	c = np.concatenate((c[1:],x))
	y_pred.append(float(x))
	y_pred_round=np.around(y_pred,decimals=6)
	absolute_error.append(float((y_true[j]-y_pred_round[j])/(y_true[j]))*100)
	sum_error+=absolute_error[j]
	j=j+1
	    
mean_error=sum_error/30
mean_error=round(mean_error,2)
mape=str(mean_error)
pred_prices=str(y_pred_round)
true_prices=str(y_true)
def execute():	
	global output
	output.config(text="True prices are " + "\n" + true_prices)  
	global output1
	output1.config(text="Predicted prices are " + "\n" + pred_prices)  
	global output2
	output2.config(text="Mean absoulute percentage error is  " + "\n" + mape + "%")
	
def graph():
	
	doge = yfinance.Ticker('USDC-USD')
	hist = doge.history(period='1y')
	yx=hist['Close']
	yx.plot(title="USDC Prices")
	plt.show()
	
def pred_graph():
	
	plt.plot(y_true, color="black", label="Actual Prices")
	plt.plot(y_pred_round, color="green", label="Predicited Prices")
	plt.title(f"{crypto_currency} Price Prediction")
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.show()
	

frames = [PhotoImage(file='usdc.png')]

label=tkinter.Label(window,image=frames).grid(row=0,column=0)

run=Button(window,text="Execute" ,bg="#2bc260" ,fg="black" ,width='100', height='2' ,font=("Times New Roman",18), command = execute)
run.grid(row=1,column=0)

show=Button(window,text="Show Graph" ,bg="tomato" ,fg="black" ,width='100', height='2' ,font=("Times New Roman",18), command = graph)
show.grid(row=2,column=0)

graph1=Button(window,text="Show Prediction vs Actual Graph" ,bg="#de84e8" ,fg="black" ,width='100', height='2' ,font=("Times New Roman",18), command = pred_graph)
graph1.grid(row=3,column=0)

output = tkinter.Label(window, width='102', fg = "black",bg="#2b97c2",height='6',font=("Times New Roman",18))
output.grid(row=4,column=0)

output1 = tkinter.Label(window, width='102', fg = "black",bg="#2b97c2",height='6',font=("Times New Roman",18))
output1.grid(row=5,column=0)

output2 = tkinter.Label(window, width='102', fg = "black",bg="#2b97c2",height='3',font=("Times New Roman",18))
output2.grid(row=6,column=0)

window.mainloop()
