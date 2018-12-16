import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
def gender_churn(x,y):
    gender = x[:, 0]
    arr = [0, 0]
    churned = [0, 0, 0, 0]
    counter = 0
    labels_1 = ['Females', 'Males']
    labels_2 = ['Females|No Churn', 'Females|Churn', 'Males|No Churn', 'Males|Churn']
    for i in gender:
        if i == 1:
            arr[0] = arr[0] + 1
            if y[counter] == 0:
                churned[0] = churned[0] + 1
            else:
                churned[1] = churned[1] + 1
        else:
            arr[1] = arr[1] + 1
            if y[counter] == 0:
                churned[2] = churned[2] + 1
            else:
                churned[3] = churned[3] + 1
        counter = counter + 1
    print(churned)
    plt.pie(arr, labels=labels_1)
    plt.title('Gender Distribution of the Customers')
    plt.axis('equal')
    plt.show()
    plt.pie(churned, labels=labels_2)
    plt.axis('equal')
    plt.show()

def senior_citizen_churn(x,y):
    senior = x[:, 1]
    arr = [0, 0]
    churned = [0, 0, 0, 0]
    counter = 0
    labels_1 = ['Young', 'Senior']
    labels_2 = ['Young|No Churn', 'Young|Churn', 'Senior|No Churn', 'Senior|Churn']
    for i in senior:
        if i == 0:
            arr[0] = arr[0] + 1
            if y[counter] == 0:
                churned[0] = churned[0] + 1
            else:
                churned[1] = churned[1] + 1
        else:
            arr[1] = arr[1] + 1
            if y[counter] == 0:
                churned[2] = churned[2] + 1
            else:
                churned[3] = churned[3] + 1
        counter = counter + 1
    print(churned)
    plt.pie(arr, labels=labels_1)
    plt.title('Age Distribution of the Customers')
    plt.axis('equal')
    plt.show()
    plt.pie(churned, labels=labels_2)
    plt.axis('equal')
    plt.show()

def partner_churn(x,y):
    senior = x[:, 2]
    arr = [0, 0]
    churned = [0, 0, 0, 0]
    counter = 0
    labels_1 = ['Partner', 'Single']
    labels_2 = ['Partner|No Churn', 'Partner|Churn', 'Single|No Churn', 'Single|Churn']
    for i in senior:
        if i == 0:
            arr[0] = arr[0] + 1
            if y[counter] == 0:
                churned[0] = churned[0] + 1
            else:
                churned[1] = churned[1] + 1
        else:
            arr[1] = arr[1] + 1
            if y[counter] == 0:
                churned[2] = churned[2] + 1
            else:
                churned[3] = churned[3] + 1
        counter = counter + 1
    print(churned)
    plt.pie(arr, labels=labels_1)
    plt.title('Age Distribution of the Customers')
    plt.axis('equal')
    plt.show()
    plt.pie(churned, labels=labels_2)
    plt.axis('equal')
    plt.show()

def tenure_churn(x,y):
    tenure = x[:,4]
    plt.plot(range(0,7043),tenure)
    plt.show()
    plt.scatter(y,tenure,color='red')
    plt.show()

# Getting the dataset

data = pd.read_csv('churn.csv')
print(data.head())
print(data.describe())
print(data.columns)
data = data.drop(labels='customerID',axis=1)
x = data.iloc[:,:-2].values
y = data.iloc[:,-1].values
y = np.reshape(y,(-1,1))

# Data Encoding
binary_encoder = LabelEncoder()
tertiary_encoder = LabelEncoder()
quadra_encoder = LabelEncoder()
x[:,0] = binary_encoder.fit_transform(x[:,0])
x[:,2] = binary_encoder.fit_transform(x[:,2])
x[:,3] = binary_encoder.fit_transform(x[:,3])
x[:,5] = binary_encoder.fit_transform(x[:,5])
x[:,6] = tertiary_encoder.fit_transform(x[:,6])
x[:,7] = tertiary_encoder.fit_transform(x[:,7])
x[:,8] = tertiary_encoder.fit_transform(x[:,8])
x[:,9] = tertiary_encoder.fit_transform(x[:,9])
x[:,10] = tertiary_encoder.fit_transform(x[:,10])
x[:,11] = tertiary_encoder.fit_transform(x[:,11])
x[:,12] = tertiary_encoder.fit_transform(x[:,12])
x[:,13] = tertiary_encoder.fit_transform(x[:,13])
x[:,14] = tertiary_encoder.fit_transform(x[:,14])
x[:,15] = binary_encoder.fit_transform(x[:,15])
x[:,16] = quadra_encoder.fit_transform(x[:,16])

y = binary_encoder.fit_transform(y)
senior_citizen_churn(x,y)
partner_churn(x,y)
gender_churn(x,y)
tenure_churn(x,y)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=1998)




lin_r = LinearRegression()
bayesian = GaussianNB()
tree = DecisionTreeClassifier(criterion='entropy',splitter='random')
forest = RandomForestClassifier(n_estimators=80)
model = []
model.append(lin_r)
model.append(bayesian)
model.append(tree)
model.append((forest))

for classifier in model:
    cross_val = cross_val_score(classifier,x_train,y_train,cv=10)
    print('Cross val',cross_val)
    print(max(cross_val))
    print('########')
print('##############################################')
for classifier in model:
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    print(accuracy_score(y_test,y_pred.round()))
    print('#######')