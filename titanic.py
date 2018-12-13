#2018-12-02 第10176/10573

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")
data_train

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

train_df = data_train.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

knn.fit(X, y)
data_test = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/test.csv")


test_df = data_test.filter(
    regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = knn.predict(test_df)
print(predictions)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(
    "/Users/lingyu/Desktop/datasets/titanic/knnpredictions.csv",
    index=False)
