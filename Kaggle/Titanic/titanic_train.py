import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score

data = pd.read_csv('/home/banzee/Desktop/ML/Kaggle/Titanic/train.csv')
data = data.drop(labels='Cabin',axis=1)

sum=[]
for age in data['Age']:
    if not np.isnan(age):
        sum.append(age)
avg = math.floor(np.average(sum))
data['Age'].fillna(value=avg,inplace=True,)

data['Embarked'].fillna(value='S',inplace=True)
print(data.isna().sum())

for columns in data.columns:
    print(np.unique(data[columns]))

data = data.drop(labels='PassengerId',axis=1)
data = data.drop(labels='Name',axis=1)
data = data.drop(labels='Ticket',axis=1)
data = data.drop(labels='Embarked',axis=1)

x = data.iloc[:,[1,2,3,4,5,6]].values
y = data.iloc[:,0].values
lb = LabelEncoder()
x[:,1] = lb.fit_transform(x[:,1])

pca = PCA()
svd = TruncatedSVD(n_components=5)
sk_x = pca.fit_transform(x)
sk_x2 = svd.fit_transform(x)
variance = pca.explained_variance_ratio_
variance2  = svd.explained_variance_ratio_
plt.plot(range(0,6), variance)
plt.plot(range(0,5), variance2)
plt.show()


pc = PCA(n_components=6)
sv = TruncatedSVD(n_components=5)
x_1 = pc.fit_transform(x)
x_2 = pc.fit_transform(x)

dt = DecisionTreeClassifier(criterion='entropy')
rf = RandomForestClassifier(n_estimators=180,criterion='entropy')
nb = GaussianNB()
sv = SVC(kernel='rbf')
nn = MLPClassifier(hidden_layer_sizes=(180,),activation='relu',solver='adam',alpha=0.0008)
model= []
model.append(dt)
model.append(rf)
model.append(nb)
model.append(sv)
model.append(nn)

for classifier in model:
    score = cross_val_score(classifier,x_1,y,cv=10)
    svc = cross_val_score(classifier,x_2,y,cv=10)
    print(np.mean(score),np.std(score))
    print(np.mean(svc),np.std(score))

sum=[]
test = pd.read_csv('/home/banzee/Desktop/ML/Kaggle/Titanic/test.csv')
print(test.isnull().sum())
for age in test['Age']:
    if not np.isnan(age):
        sum.append(age)
avg = math.floor(np.average(sum))
test['Age'].fillna(value=avg,inplace=True)
test['Embarked'].fillna(value='S',inplace=True)
test['Fare'].fillna(value='26',inplace=True)
test = test.drop(labels='Cabin',axis=1)

test = test.drop(labels='PassengerId',axis=1)
test = test.drop(labels='Name',axis=1)
test = test.drop(labels='Ticket',axis=1)
test = test.drop(labels='Embarked',axis=1)
test['Sex'] = lb.fit_transform(test['Sex'])
rf.fit(x,y)
result = rf.predict(test)
print(len(result))
