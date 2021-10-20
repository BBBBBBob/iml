import numpy as np
import pandas as pd

train_features = pd.read_csv('./train_features.csv')
test_features = pd.read_csv('./test_features.csv')
train_features = np.array(train_features)
test_features = np.array(test_features)


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
    print('filling')
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


new_train_features = generateFeatures(train_features)
new_train_features = pd.DataFrame(data=new_train_features)
new_test_features = generateFeatures(test_features)
new_test_features = pd.DataFrame(data=new_test_features)
print('finish')
new_train_features.to_csv('new_train_features_3.csv', index=False)
new_test_features.to_csv('new_test_features_3.csv', index=False)


