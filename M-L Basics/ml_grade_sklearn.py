import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ML.csv',skipfooter=11,engine='python')
data = data.dropna()
X= data[['Year','Attendance %','M/F','CGPA','Mid Sem Grades','Quiz1 (30)']].values
X_mid_quiz = data[['Mid Sem Grades','Quiz1 (30)']].values
print(data)
year = data['Year'].values
grade = data['Grade'].values
gender = data.iloc[:,3].values
quiz = data['Quiz2 (30)'].values
mid_sem_marks = data['Mid Semester'].values
from sklearn.preprocessing import LabelEncoder
lb_grade = LabelEncoder()
lb_gender = LabelEncoder()
grade = lb_grade.fit_transform(grade)
gender = lb_gender.fit_transform(gender)
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean', missing_values='NaN', axis=1)
year = imp.fit_transform([year])
grade = np.array(grade)
year = np.array(year)
cov_year = np.corrcoef(grade,year)


print('The covariance coeff of study and final grade is',cov_year[0][1])

mid_sem_grade = data.iloc[:,6].values
mid_sem_grade = lb_grade.fit_transform(mid_sem_grade)
mid_sem_grade = np.array(mid_sem_grade)
cov_midsem = np.corrcoef(mid_sem_grade,grade)

print(np.corrcoef(quiz, mid_sem_marks))


cgpa= data.iloc[:,4].values
cov_cgpa = np.corrcoef(cgpa, grade)
plt.scatter(cgpa,grade)
plt.show()
print('The covariance coeff of cgpa and final grade is', cov_cgpa[0][1])

attendance = data.iloc[:,2].values
cov_attendance = np.corrcoef(attendance, grade)
plt.scatter(attendance,grade)
plt.show()
print('The covariance coeff of attendance and final grade is', cov_attendance[0][1])

cov_gender = np.corrcoef(gender,grade)
print('The covariance coeff of gender and final grade is', cov_gender[0][1])
plt.scatter(gender,grade)
plt.show()

# Encoding the dataset

X[:,2],X[:,4] = gender, mid_sem_grade

# Splitting Dataset

from sklearn.cross_validation import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, grade, train_size=0.8, test_size=0.2, random_state=1998)

# Without PCA
from sklearn.tree import DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
decision_tree_classifier.fit(x_train,y_train)
prediction_decision_tree = decision_tree_classifier.predict(x_test)
plt.scatter(range(0,15), y_test, label = 'True Values', marker='x')
plt.scatter(range(0,15), prediction_decision_tree, label = 'Values Predicted via Decision Tree')
plt.xlabel('Test_cases')
plt.ylabel('Corresponding Encoded Grades')
plt.legend(loc='best')
plt.show()
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error

print('Accuracy of Tree Before PCA:',accuracy_score(y_test,prediction_decision_tree))
print('Precision of Tree before PCA:',precision_score(y_test, prediction_decision_tree, average='weighted'))

from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
bayes.fit(x_train,y_train)
prediction_naive_bayes = bayes.predict(x_test)
plt.scatter(range(0,15), y_test, label = 'True Values',marker='x')
plt.scatter(range(0,15), prediction_naive_bayes, label = 'Values Predicted via Naive Bayes')
plt.xlabel('Test_cases')
plt.ylabel('Corresponding Encoded Grades')
plt.legend(loc='best')
plt.show()

print('Accuracy of Naive Bayes before PCA:',accuracy_score(y_test,prediction_naive_bayes))
print('Precision Score of Naive Bayes before PCA;',precision_score(y_test, prediction_naive_bayes, average='weighted'))

# Implementing PCA

from sklearn.decomposition import PCA
pc_check = PCA(n_components=None)
x_train_for_pca = x_train
x_test_for_pca = x_test
x_train_for_pca = pc_check.fit_transform(x_train)
x_test_for_pca = pc_check.fit_transform(x_test)
individual_variance = pc_check.explained_variance_ratio_
cumulative_variance = np.cumsum((individual_variance))
plt.plot(range(0,6),individual_variance,label='Individual Explained Variance')
plt.step(range(0,6),cumulative_variance,label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('No of parameters')
plt.legend(loc='best')
plt.show()

pc_final = PCA(n_components=2)
x_train = pc_final.fit_transform(x_train)
x_test = pc_final.transform(x_test)

# Implementing Models


decision_tree_classifier_after_pca = DecisionTreeClassifier(criterion='entropy', random_state=0)
decision_tree_classifier_after_pca.fit(x_train, y_train)
prediction_decision_tree = decision_tree_classifier_after_pca.predict(x_test)
plt.scatter(range(0,15), y_test, label = 'True Values', marker='x')
plt.scatter(range(0,15), prediction_decision_tree, label = 'Values Predicted via Decision Tree after PCA')
plt.xlabel('Test_cases')
plt.ylabel('Corresponding Encoded Grades')
plt.legend(loc='best')
plt.show()
print('Accuracy Score Decision Tree after PCA:',accuracy_score(y_test, prediction_decision_tree))
print('Precision Score Decision Tree after PCA', precision_score(y_test, prediction_decision_tree, average='weighted'))

naive_bayes_after_pca = GaussianNB()
naive_bayes_after_pca.fit(x_train,y_train)
prediction_naive_bayes = naive_bayes_after_pca.predict(x_test)
plt.scatter(range(0,15), y_test, label = 'True Values',marker='x')
plt.scatter(range(0,15), prediction_naive_bayes, label = 'Values Predicted via Naive Bayes after PCA')
plt.xlabel('Test_cases')
plt.ylabel('Corresponding Encoded Grades')
plt.legend(loc='best')
plt.show()
print('Accuracy Score Naive Bayes after PCA:',accuracy_score(y_test,prediction_naive_bayes))
print('Precision Score Naive Bayes after PCA:', precision_score(y_test, prediction_naive_bayes, average='weighted'))