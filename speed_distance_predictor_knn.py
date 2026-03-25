# library imports
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np 

# data set reading
mydata = pd.read_csv("speed_distance.csv")
x = mydata[["X"]]
y = mydata[["Y"]]

# model creation and training
model = KNeighborsRegressor(n_neighbors = 3)
model.fit(x,y)

#predicting new value
new_distance = model.predict([[160]])
print("Predicted distance = ", new_distance)

#model evaluation
y_pred = model.predict(x)
mse = mean_squared_error(y,y_pred)
print("MSE = ", mse)
rmse = np.sqrt(mse)
print("RMSE = ", rmse)

#visualization
plt.scatter(x,y)
plt.plot(x,y_pred, color = 'red')
plt.show()