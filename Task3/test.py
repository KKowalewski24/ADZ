import matplotlib.pyplot as plt
from pmdarima import auto_arima

from module.reader import read_air_passengers, read_alcohol_sales, read_gold_price

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
