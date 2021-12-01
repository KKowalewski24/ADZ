import numpy as np
import matplotlib.pyplot as plt
from module.reader import read_air_passengers, read_gold_price, read_alcohol_sales
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima

# read datasets
passengers, passengers_outliers = read_air_passengers()
gold, gold_outliers = read_gold_price()
alcohol, alcohol_outliers = read_alcohol_sales()

x = passengers.iloc[:, 1]
x = gold.iloc[:, 1]
x = alcohol.iloc[:, 1]

# search ARIMA model
model = auto_arima(x)
pred = model.predict_in_sample()
plt.plot(x)
plt.plot(pred)
plt.show()
