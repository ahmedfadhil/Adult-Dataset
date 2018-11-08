import numpy as np
import pandas as pd
# load sklearn and encode all object type variables
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# check data set
train.info()

print("The train data has: ", train.shape)
print("The test data has: ", test.shape)

# Let have a glimpse of the data set
train.head()

# Now, let's check the missing values (if present) in this data.
nans = train.shape[0] - train.dropna().shape[0]
print("%d rows have missing values in the train data" % nans)
nand = test.shape[0] - test.dropna().shape[0]
print("%d rows have missing values in the test data" % nand)

# Which columns have missing values
train.isnull().sum()

cat = train.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)

# Education
train.workclass.value_counts(sort=True)
train.workclass.fillna('Private', inplace=True)
# Occupation
train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty', inplace=True)
# Native Country
train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States', inplace=True)

train.target.value_counts() / train.shape[0]
pd.crosstab(train.education, train.target, margins=True) / train.shape[0]

# load sklearn and encode all object type variables
for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))

train.target.value_counts()

# Building a Random Forest Model
y = train['target']

del train['target']


X = train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

# Train the RF classifier
clf = RandomForestClassifier(n_estimators=500, max_depth=6)
clf.fit(X_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=6, max_features='auto',
                       max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
clf.predict(X_test)

# make prediction and check model's accuracy
prediction = clf.predict(X_test)
accuracy = accuracy_score(np.array(y_test), prediction)
print('The accuracy of Random Forest is {}'.format(accuracy))

