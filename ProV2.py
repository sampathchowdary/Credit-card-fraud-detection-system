import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools

print "Started"
#reading data
data= pd.read_csv("creditcard.csv")
f0=open('fraud.txt','a')
f1=open('genuine.txt','a')

sc= StandardScaler()
data["scaled_Amount"]=sc.fit_transform(data["Amount"].values.reshape(-1,1))

#dropping time and old amount column
data= data.drop(["Time","Amount","V8","V13","V15","V20","V22","V23","V24","V25","V26","V27","V28"], axis=1)

f_cnt= len(data[data["Class"]==1])
n_cnt = len(data[data["Class"]==0])

#getting fraud and normal transaction indexes
fraud_index= np.array(data[data["Class"]==1].index)
normal_index= data[data["Class"]==0].index

#choosing random normal indices equal to the number of fraudulent transactions
ind= np.random.choice(normal_index, f_cnt, replace= False)
ind= np.array(ind)

# concatenate fraud index and normal index to create a list of indices
ui= np.concatenate([fraud_index, ind])

#use the undersampled indices to build the ud dataframe
ud= data.iloc[ui, :]
X= ud.iloc[:, ud.columns != "Class"].values
y= ud.iloc[:, ud.columns == "Class"].values

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y, test_size= 0.25, random_state= 0)

#separating the x and y variables to fit our model
X_full= data.iloc[:, ud.columns != "Class"].values
y_full= data.iloc[:, ud.columns == "Class"].values

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.25, random_state = 0)

print "Training Started"
print "Logistic Regression"
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train.ravel())
y_pred_lr=lr.predict(X_test)
y_pred_prob_lr = lr.predict_proba(X_test)

print "Decision tree"
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt=lr.predict(X_test)
y_pred_prob_dt=dt.predict_proba(X_test)

print "Naive Bayesian"
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb=gnb.predict(X_test)
y_pred_prob_gnb =gnb.predict_proba(X_test)

print "SVC"
svc =  SVC(C= 1, kernel="linear", random_state= 0,probability=True)
svc.fit(X_train_u, y_train_u.ravel())
y_pred_svc= svc.predict(X_test)
y_pred_prob_svc=svc.predict_proba(X_test)

print "Predicting Started"
y_pred=[]
l=len(y_pred_prob_lr)
for i in range(l):
        m=(y_pred_prob_lr[i][0]+y_pred_prob_gnb[i][0]+y_pred_prob_svc[i][0])/3
        if m>0.75:
            y_pred.append(0)
            f1.write(','.join(map(str,X_test[i]))+",0" + "\n")
        elif m<0.2:
            y_pred.append(1)
            f0.write(','.join(map(str,X_test[i])) +",1" + "\n")
        else:
            r = X_test[i]
            v=dt.predict(r.reshape(1,-1))
            y_pred.append(v)
            if v==0:
                f1.write(','.join(map(str,X_test[i]))+",0" + "\n")
            else:
                f0.write(','.join(map(str,X_test[i])) +",1" + "\n")

CM = metrics.confusion_matrix(y_test, y_pred)
print "Confusion Martix\n",CM
acc=float(CM[1][1]+CM[0][0])/(CM[0][0] + CM[0][1]+CM[1][0] + CM[1][1])*100

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, cmap=plt.cm.Pastel1)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="black")
    plt.show()

plt.figure()
plot_confusion_matrix(CM, classes="0,1")
print "Accuracy of proposed model:",acc

print "Completed"
