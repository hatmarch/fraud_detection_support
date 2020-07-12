# In[4]:


import joblib

### From seprate part of notebook
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

print("Reading csv data")
df = pd.read_csv ("data/creditcard.csv")
# rename specific columns
df.rename(columns={'Unnamed: 0':'Id'}, inplace=True)

print("Data loaded using File.")


# shuffle the rows
df = df.sample(frac=1).reset_index()

#Order the credit card transaction by transaction time
#df.sort_values(by=['Time'])

#number of rows in the dataset
n_samples = df.shape[0]
print("Samples: %d" % n_samples)

#Split into train and test
train_size = 0.75

train_limit = int(n_samples * train_size)
df_train = df.iloc[:train_limit]
df_test = df.iloc[train_limit:]

### END separate part

#Define features and target variables.

# Choose either the features definition based on important features or correlation

## From the important features graph we only want seven important features:
##   V3,V4,V10,V11,V12,V14,V17
features = ['V3','V4','V10','V11','V12','V14','V17']

## using the correllation graph, select the top seven important features:
## V14,V17,V3,V12,V10,V16,V11
#features = ['V14','V17','V3','V12','V10','V16','V11']

non_features = [i for i in df_train.columns if i not in features]
class_column=['Class']


features_train = df_train.drop(non_features, axis=1)
target_train = df_train.loc[:, "Class"]

features_test = df_test.drop(non_features, axis=1)
target_test = df_test.loc[:, "Class"]
print("feature_test columns:")
print(features_test.columns)

model = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=10, class_weight='balanced')

model.fit(features_train, target_train.values.ravel())

pred_train = model.predict(features_train)
pred_test = model.predict(features_test)

pred_train_prob = model.predict_proba(features_train)
pred_test_prob = model.predict_proba(features_test)

print("Number of features")
print(len(model.feature_importances_))
  
#save mode in filesystem
joblib.dump(model, 'model.pkl') 


