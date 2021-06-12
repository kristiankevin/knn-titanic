#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv') #dataset
df = df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)
df.info()
#preprocess data
#Replace Nan age w/ median
avg_age = df['Age'].astype("float").median(axis=0)
df['Age'].replace(np.nan, avg_age, inplace=True)
#Label Encoder for Sex m/f=1/0
df['Sex'] = preprocessing.LabelEncoder().fit_transform(df['Sex'])

#DATA NORMALIZATION (StandardScaler)
X_row = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df[X_row] = preprocessing.StandardScaler().fit(df[X_row]).transform(df[X_row].astype(float))

X = df.iloc[:, 1:]
y = df.iloc[:,0]

#validation split stratifiedkfold
skf = StratifiedKFold(n_splits=5)
#training model (KNN)
knn_model = KNeighborsClassifier(n_neighbors = 5)
acc_score = []

#skf val fit
for train_index, test_index in skf.split(X, y):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

    knn_model.fit(X_train,y_train)
    pred_values = knn_model.predict(X_test)

    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)


avg_acc_score = sum(acc_score)/len(acc_score) #avg score

#FINAL SCORE
print('accuracy of each fold — {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

#testset run
df_test = pd.read_csv('test.csv')
id = df_test.loc[:, 'PassengerId'] #Save PassengerId
df_test = df_test.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1) #drop other column
df_test.info()

#preprocess data
med_age_test = df_test['Age'].astype("float").median(axis=0)
df_test['Age'].replace(np.nan, med_age_test, inplace=True)

df_test['Sex'] = preprocessing.LabelEncoder().fit_transform(df_test['Sex'])

#Fix Lost Fare with Median of Its Pclass
df_ps_avg_test = df_test[['Pclass','Fare']].groupby(['Pclass'],as_index=False).median()
df_test.loc[df_test['Fare'].isnull(), 'Fare']= df_ps_avg_test.iloc[2]['Fare']

#DATA NORMALIZATION (StandardScaler)
df_test[X_row] = preprocessing.StandardScaler().fit(df_test[X_row]).transform(df_test[X_row].astype(float))

yHat = knn_model.predict(df_test.values)
yHat[:10]
df_result = pd.DataFrame(id)
df_result['Survived'] = yHat

df_result.info()

df_result.to_csv('result2.csv', index=False)
