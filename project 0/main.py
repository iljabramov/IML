import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv").to_numpy()
test = pd.read_csv("test.csv").to_numpy()
solution = np.sum(test[:, 1:], axis= 1) / 10
test = test[:, 1:]
inputs = train[:,2:]
targets = train[:, 1]
model = LinearRegression()
model.fit(inputs, targets)
pred = model.predict(test)
rmse = mean_squared_error(solution, pred) ** 0.5
print(rmse)
pred = pd.DataFrame(
    {
        "Id": np.arange(10000, 10000 + len(pred)),
        "y" : pred
    })
pred.to_csv("my.csv", index= False)

