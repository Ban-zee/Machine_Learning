''' Data Generation
**********************************************************************************************************************************************'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pc
import seaborn as sns

x = np.random.rand(500)
test = x
noise = np.random.normal(0.011,0.4,500)
X = pd.DataFrame(data=x)
sk_test = X.iloc[:,0].values
X['Intercept']=1
y=[]
w_0 = -0.09*25
w_1 = 0.2*25
for i in range(0,500):
    c = w_0+x[i]*w_1+noise[i]
    y.append(c)
y = np.array(y)
Y = pd.DataFrame(data=y)
'''*******************************************************************************************************************************************'''

# Linear Basis with f(x) = x as the basis function

x = X.iloc[:,[0,1]].values
y = Y.iloc[:,0].values
z = np.matmul(np.transpose(x),x)
z = np.linalg.inv(z)
z = np.matmul(z,np.transpose(x))
coefficients = np.matmul(z,y)
slope = float(coefficients[0])
intercept = float(coefficients[1])
y_pred=[]
y_true=[]
for i in range (0,500):
    c = intercept+float(test[i])*slope
    d = w_0+float(test[i])*w_1
    y_pred.append(c)
    y_true.append(d)

"""SK-LEARN LINEAR REG************************************************************************************************************************"""
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
sk_test = np.reshape(X.iloc[:,0].values,(-1,1))
sk_y = np.reshape(X.iloc[:,0].values,(-1,1))
reg.fit(sk_test,sk_y)
y_sk_pred = reg.predict(sk_test)
'''*******************************************************************************************************************************************'''

# Bayesian Linear Regression
model = pc.Model()
with model as linear_model:
    I = pc.Normal('I', mu=0, sd=10)
    S = pc.Normal('S', mu=0, sd=10)
    SD = pc.HalfNormal('SD', sd=10)
    mean = I+S*X.iloc[0:20,0].values
    y_bayes_pred = pc.Normal('y_bayes_pred', mu = mean, sd=SD, observed = Y.iloc[0:20,0].values)
    step = pc.NUTS()
    linear_trace = pc.sample(100,step)
pc.plot_posterior_predictive_glm(linear_trace, samples=200, eval=test,color='green',alpha=0.7, linewidth=1, lm=lambda x, sample:sample['I']+sample['S']*x, label='Bayesian Linear Regression')
plt.scatter(X[0],Y,color='pink')
plt.plot(X[0],y_pred, label='MLE Linear Regression',color='orange')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.show()

bayes_prediction = linear_trace['I']+ linear_trace['S']*0.8158
sns.kdeplot(bayes_prediction, label='Bayesian Estimate')
plt.vlines(x=0.8158*slope+intercept,color='orange',ymin=0, ymax=2.6, label='MLE Estimate')
plt.xlabel('Predicted value of Y')
plt.ylabel('Probability Density')
plt.legend(loc='best')
plt.show()