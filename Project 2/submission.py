import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def generateFeatures(Features):
    num_features = Features.shape[1] - 3
    num_patient = int(Features.shape[0] / 12)

    new_features = np.zeros((num_patient, num_features * 5 + 2))

    for i in range(num_patient):
        feature_patient = Features[12*i: 12*i+12, 3:]
        new_features[i, 0] = Features[12*i, 0]
        new_features[i, 1] = Features[12*i, 2]
        for j in range(num_features):
            one_feature = feature_patient[:, j]
            new_features[i, 5*j+2] = np.nanmean(one_feature, axis=0)
            new_features[i, 5*j+3] = np.nanmedian(one_feature, axis=0)
            new_features[i, 5*j+4] = np.nanstd(one_feature, axis=0)
            new_features[i, 5*j+5] = np.nanmax(one_feature, axis=0)
            new_features[i, 5*j+6] = np.nanmin(one_feature, axis=0)

    age = new_features[:, 1]

    for i in range(num_patient):
        similar_age = np.argwhere((age <= age[i] + 10) & (age >= age[i] - 10))
        find_nan = np.argwhere(np.isnan(new_features[i, :]))
        similar_age = np.squeeze(similar_age, axis=1)
        find_nan = np.squeeze(find_nan, axis=1)
        num_nan = np.size(find_nan)
        num_age = np.size(similar_age)
        if num_nan > 0:
            if num_age == 1:
                new_features[i, find_nan] = 0
            else:
                similar_features = new_features[similar_age, :]
                nan_features = similar_features[:, find_nan]
                new_features[i, find_nan] = np.nanmedian(nan_features, axis=0)
                find_nan = np.argwhere(np.isnan(new_features[i, :]))
                find_nan = np.squeeze(find_nan, axis=1)
                num_nan = np.size(find_nan)
                if num_nan > 0:
                    new_features[i, find_nan] = 0

    return new_features


train_features = pd.read_csv('./train_features.csv')
test_features = pd.read_csv('./test_features.csv')
train_labels = pd.read_csv('./train_labels.csv')
train_features = np.array(train_features)
test_features = np.array(test_features)

new_train_features = generateFeatures(train_features)
new_test_features = generateFeatures(test_features)

X_train = new_train_features[:, 1:]
Y_train = np.array(train_labels)
headers = train_labels.columns.values
Y_train1 = Y_train[:, 1:12]
Y_train2 = Y_train[:, 12:16]
X_test = new_test_features[:, 1:]
result = np.zeros((X_test.shape[0], Y_train.shape[1]))
result[:, 0] = new_test_features[:, 0]

forest = RandomForestClassifier(n_estimators=800, max_depth=14)
clf = MultiOutputClassifier(forest, n_jobs=-1).fit(X_train, Y_train1)
prob = clf.predict_proba(X_test)

for i in range(len(prob)):
    result[:, i+1] = prob[i][:, 1]

coefList = np.array([10, 50, 200, 500, 1000, 2000, 5000, 10000])
i = 0
scorelist = np.zeros((coefList.size, 1))

for coef in coefList:
    ridge = Ridge(alpha=coef, fit_intercept=True)
    ridge.fit(X_train, Y_train2)
    scores = cross_val_score(ridge, X_train, Y_train2, scoring="neg_root_mean_squared_error", cv=10)
    scorelist[i] = np.mean(-scores)
    i += 1

opt_coef = np.argmin(scorelist)
clf = Ridge(alpha=coefList[opt_coef], fit_intercept=True)
clf.fit(X_train, Y_train2)
result[:, 12:16] = clf.predict(X_test)

submission = pd.DataFrame(data=result, columns=headers)
submission.to_csv('submission.csv', index=False)
submission.to_csv('submission_data.zip', index=False, float_format='%.3f', compression='zip')
print('finish')

