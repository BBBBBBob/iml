import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

k = 10

datafile = pd.read_csv('./train.csv')
data = np.array(datafile)

lambdaList = np.array([0.1, 1, 10, 100, 200])
cvList = np.zeros((lambdaList.size, 1))


y = data[:, 0]
x = data[:, 1:]

kf = KFold(n_splits=k, shuffle=False)

i = 0
for rlambda in lambdaList:
    Sum = 0
    for train_index, test_index in kf.split(x):
        x_train = x[train_index, :]
        x_test = x[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]

        rr = Ridge(alpha=rlambda)
        rr.fit(x_train, y_train)
        y_pred = rr.predict(x_test)
        RMSE = mean_squared_error(y_test, y_pred) ** 0.5
        Sum += RMSE

    CV = Sum/k
    cvList[i] = CV
    i += 1

np.savetxt('submission.csv', cvList)