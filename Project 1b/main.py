import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

datafile = pd.read_csv('./train.csv')
data = np.array(datafile)

lambdaList = np.array([0.1, 1, 10, 100, 200])
cvList = np.zeros((lambdaList.size, 1))

k = 10
weight = []

y = data[:, 1]
x = data[:, 2:]

x_feature = np.hstack((x, x ** 2))
x_feature = np.hstack((x_feature, np.exp(x)))
x_feature = np.hstack((x_feature, np.cos(x)))
x_feature = np.hstack((x_feature, np.ones([len(y), 1])))

kf = KFold(n_splits=k, shuffle=False)

i = 0
threshold = float('inf')

for rlambda in lambdaList:
    Sum = 0
    for train_index, test_index in kf.split(x_feature):
        x_train = x_feature[train_index, :]
        x_test = x_feature[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]

        krg = Ridge(alpha=rlambda, fit_intercept=False)
        krg.fit(x_train, y_train)
        y_pred = krg.predict(x_test)
        RMSE = mean_squared_error(y_test, y_pred) ** 0.5
        Sum += RMSE

    CV = Sum/k

    if CV < threshold:
        rg = Ridge(alpha=rlambda, fit_intercept=False)
        rg.fit(x_feature, y)
        weight = rg.coef_
        threshold = CV

np.savetxt('submissionJiachengQiu.csv', weight)

