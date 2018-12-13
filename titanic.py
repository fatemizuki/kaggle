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

#2018-12-03 0.6555
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")

train_df = data_train.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

data_test = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/test.csv")
test_df = data_test.filter(
    regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

from sklearn.neighbors import KNeighborsClassifier
# n_neighbours = [1,2,3,4,5,6,7]

# for n_neighbours in n_neighbours:
#
#     knn = KNeighborsClassifier(n_neighbors=n_neighbours)
#     knn.fit(X_train, y_train)
#     print("n_neighbours:{},train:{:.2f}".format(n_neighbours,knn.score(X_train,y_train)))
#     print("n_neighbours:{},test:{:.2f}".format(n_neighbours,knn.score(X_test,y_test)))

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
predictions = knn.predict(test_df)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(
    "/Users/lingyu/Desktop/datasets/titanic/knnpredictions3.csv",
    index=False)


#2018-12-04 9966 0.69856
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")

train_df = data_train.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

data_test = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/test.csv")
test_df = data_test.filter(
    regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

from sklearn.tree import DecisionTreeClassifier
# max_depth = [3,4,5,6,7]
#
# for max_depth in max_depth:
#
#     log = DecisionTreeClassifier(max_depth= max_depth,random_state=7)
#     log.fit(X_train, y_train)
#     print("n_neighbours:{},train:{:.2f}".format(max_depth,log.score(X_train,y_train)))
#     print("n_neighbours:{},test:{:.2f}".format(max_depth,log.score(X_test,y_test)))

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train, y_train)
predictions = dtc.predict(test_df)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(
    "/Users/lingyu/Desktop/datasets/titanic/treepredictions.csv",
    index=False)

#9849 0.71291
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")

train_df = data_train.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass')
train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

data_test = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/test.csv")
data_test = data_test.fillna(data_test.mean())
test_df = data_test.filter(
    regex='Age_.*|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

from sklearn.tree import DecisionTreeClassifier
# max_depth = [3,4,5,6,7]
#
# for max_depth in max_depth:
#
#     log = DecisionTreeClassifier(max_depth= max_depth,random_state=7)
#     log.fit(X_train, y_train)
#     print("n_neighbours:{},train:{:.2f}".format(max_depth,log.score(X_train,y_train)))
#     print("n_neighbours:{},test:{:.2f}".format(max_depth,log.score(X_test,y_test)))

dtc = DecisionTreeClassifier(max_depth=6)
dtc.fit(X_train, y_train)
predictions = dtc.predict(test_df)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(
    "/Users/lingyu/Desktop/datasets/titanic/treepredictions2.csv",
    index=False)

#0.75119 9034
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split


data_train = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")
data_train = data_train.fillna(data_train.mean())
sex_transfer = {'male':1,'female':2}
data_train['Sex'] = data_train['Sex'].map(sex_transfer)
train_df = data_train.filter(
    regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex|Pclass')

train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

data_test = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/test.csv")
data_test = data_test.fillna(data_test.mean())
data_test['Sex'] = data_test['Sex'].map(sex_transfer)
test_df = data_test.filter(
    regex='Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex|Pclass')

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7)

from sklearn.ensemble import RandomForestClassifier
# max_depth = [9,10,11,12,13]
#
# for max_depth in max_depth:
#
#     forest = RandomForestClassifier(n_estimators=1000,max_depth=max_depth,random_state=7)
#     forest.fit(X_train, y_train)
#     print("max_depth:{},train:{:.2f}".format(max_depth,forest.score(X_train,y_train)))
#     print("max_depth:{},test:{:.2f}".format(max_depth,forest.score(X_test,y_test)))

forest = RandomForestClassifier(n_estimators=7000,max_depth=10,random_state=7)
forest.fit(X_train, y_train)
predictions = forest.predict(test_df)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(
    "/Users/lingyu/Desktop/datasets/titanic/rtpredictions.csv",
    index=False)


#0.77590 5316
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split


data_train = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/train.csv")
data_train = data_train.fillna(data_train.mean())
sex_transfer = {'male': 1, 'female': 2}
data_train['Sex'] = data_train['Sex'].map(sex_transfer)
train_df = data_train.filter(
    regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex|Pclass')

train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

data_test = pd.read_csv("/Users/lingyu/Desktop/datasets/titanic/test.csv")
data_test = data_test.fillna(data_test.mean())
data_test['Sex'] = data_test['Sex'].map(sex_transfer)
test_df = data_test.filter(
    regex='Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex|Pclass')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=7)

from sklearn.ensemble import RandomForestClassifier
# max_depth = [9,10,11,12,13]
#
# for max_depth in max_depth:
#
#     forest = RandomForestClassifier(n_estimators=1000,max_depth=max_depth,random_state=7)
#     forest.fit(X_train, y_train)
#     print("max_depth:{},train:{:.2f}".format(max_depth,forest.score(X_train,y_train)))
#     print("max_depth:{},test:{:.2f}".format(max_depth,forest.score(X_test,y_test)))

forest = RandomForestClassifier(
    n_estimators=50000,
    max_depth=6,
    n_jobs=-1,
    random_state=7)
forest.fit(X_train, y_train)
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))
predictions = forest.predict(test_df)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(
    "/Users/lingyu/Desktop/datasets/titanic/rtpredictions5.csv",
    index=False)
