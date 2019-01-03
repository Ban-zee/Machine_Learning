import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
dataset = pd.read_csv('churn.csv')
missing_value_row = list(dataset[dataset['TotalCharges'] == " "].index)
for missing_row in missing_value_row :
    dataset['TotalCharges'][missing_row] = 0
x = dataset.iloc[:,1:20].values
y = dataset.iloc[:,20].values


from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
y = l.fit_transform(y)
l_x = LabelEncoder()
l_p = LabelEncoder()
x[:,0] = l_x.fit_transform(x[:,0])
x[:,2] = l_p.fit_transform(x[:,2])
x[:,3] = l_p.fit_transform(x[:,3])
for i in range(5,17):
    x[:, i] = l_p.fit_transform(x[:, i])

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.01, random_state=2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA
pc = PCA(n_components=2)
x_train = pc.fit_transform(x_train)
x_test = pc.transform(x_test)
variance = pc.explained_variance_ratio_

from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
z = confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
