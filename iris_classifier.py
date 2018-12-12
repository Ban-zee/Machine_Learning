import pandas as pd

dataset = pd.read_csv('Iris.csv',header=0)
print(dataset.describe())
y = dataset.iloc[:,5].values
x = dataset.iloc[:,1:5].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2001)

from sklearn.decomposition import KernelPCA
pc = KernelPCA(n_components=2, kernel='rbf')
x_test = pc.fit_transform(x_test)
x_train = pc.transform(x_train)

from sklearn.tree import DecisionTreeClassifier
predictor = DecisionTreeClassifier(criterion='entropy')
predictor.fit(x_train,y_train)
y_pred = predictor.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
forrest = RandomForestClassifier(n_estimators=2, criterion='entropy')
forrest.fit(x_train, y_train)
forest_pred = forrest.predict(x_test)

from sklearn.svm import SVC
sv = SVC(kernel='rbf')
sv.fit(x_train,y_train)
svm_pred = sv.predict(x_test)

from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
bayes.fit(x_train,y_train)
bayes_pred = bayes.predict(x_test)

from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier()
boost.fit(x_train,y_train)
boost_pred = boost.predict(x_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=1998)
log_reg.fit(x_train,y_train)
reg = log_reg.predict(x_test)



from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, forest_pred))
print(accuracy_score(y_test, svm_pred))
print(accuracy_score(y_test, bayes_pred))
print(accuracy_score(y_test, boost_pred))
print(accuracy_score(y_test, reg))


