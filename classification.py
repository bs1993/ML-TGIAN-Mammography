
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn import metrics

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)

from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot

#fortosi twn dedomenwn apo to csv
mydata = pd.read_csv('mammography.csv')

# #visualize to plithos twn eggrafwn ana katigoria
import seaborn as sns
sns.countplot(mydata['result'],label="Count")
plt.show()




# ##kratao ta onomata ton   arxikwn steilon pou tha xrisimopoiisoume sto classification
feature_names = ['area', 'gray_level', 'gradient_strength', 'fluctuation', 'contrast','low_order_moment']
X = mydata[feature_names] #ftiaxnomai ton pinaka pou tha exei tis stiles
y = mydata['result'] #ta labels ton stilon

#sxediazoume to scatter matrix
pd.plotting.scatter_matrix(X)

#apo tin grammi 49 mexri tin grammi 62
#ypologizetai to correlation metaksy twn xaraktiristikwn
#kai apomakrynontai ayta poy exoun correlation panw apo 0.9
label_encoder = LabelEncoder()
X.iloc[:,0] = label_encoder.fit_transform(X.iloc[:,0]).astype('float64')

corr = X.corr()


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
X = X[selected_columns]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7) #spasimo se train kai test
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  #kanikopoiisi dedomeon
X_test = scaler.transform(X_test) #kanikopoiisi dedomeon

# #logistic regr
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train) #ekpaideusi monteloy


matrix = plot_confusion_matrix(logreg, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our  Logistic Regression')
plt.show(matrix)
plt.show()


predicted_log  = logreg.predict(X_test);

desired = y_test.values



tp=0
fp=0
tn=0
fn=0

for i in range(len(desired)):
    if desired[i]=="'1'" and predicted_log[i]=="'1'":
        tp=tp+1
    elif desired[i]=="'1'" and predicted_log[i]=="'-1'":
        fn=fn+1
    elif desired[i]=="'-1'" and predicted_log[i]=="'-1'":
        tn=tn+1
    else:
        fp=fp+1
print("Results for Logistic Regression:")
print("True Positive:",tp)
print("False Positive:",fp)
print("True Negative:",tn)
print("False Negative:",fn)
precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp + tn + fp + fn)

f1= 2*(precision*recall)/(precision+recall)

specificity = (tn)/(tn+fp)


print("Precision:",precision)
print("Recall:",recall)
print("Accuracy:",accuracy)
print("F1 score:",f1)
print("Specificity:",specificity)

print("---------------")



y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label="'1'")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logistic Regression, auc="+str(auc))
plt.legend(loc=4)
plt.show()



# #dec tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)


predicted_dc = clf.predict(X_test)

matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our  Decision Tree')
plt.show(matrix)
plt.show()



tp=0
fp=0
tn=0
fn=0

for i in range(len(desired)):
    if desired[i]=="'1'" and predicted_dc[i]=="'1'":
        tp=tp+1
    elif desired[i]=="'1'" and predicted_dc[i]=="'-1'":
        fn=fn+1
    elif desired[i]=="'-1'" and predicted_dc[i]=="'-1'":
        tn=tn+1
    else:
        fp=fp+1
print("Results for Decision Tree:")
print("True Positive:",tp)
print("False Positive:",fp)
print("True Negative:",tn)
print("False Negative:",fn)

precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp + tn + fp + fn)

f1= 2*(precision*recall)/(precision+recall)

specificity = (tn)/(tn+fp)


print("Precision:",precision)
print("Recall:",recall)
print("Accuracy:",accuracy)
print("F1 score:",f1)
print("Specificity:",specificity)

print("---------------")


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label="'1'")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Decision Tree, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# #K-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


predicted_knn=knn.predict(X_test)

matrix = plot_confusion_matrix(knn, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our  KNN')
plt.show(matrix)
plt.show()


tp=0
fp=0
tn=0
fn=0

for i in range(len(desired)):
    if desired[i]=="'1'" and predicted_knn[i]=="'1'":
        tp=tp+1
    elif desired[i]=="'1'" and predicted_knn[i]=="'-1'":
        fn=fn+1
    elif desired[i]=="'-1'" and predicted_knn[i]=="'-1'":
        tn=tn+1
    else:
        fp=fp+1

print("Results for KNN:")
print("True Positive:",tp)
print("False Positive:",fp)
print("True Negative:",tn)
print("False Negative:",fn)

precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp + tn + fp + fn)

f1= 2*(precision*recall)/(precision+recall)

specificity = (tn)/(tn+fp)


print("Precision:",precision)
print("Recall:",recall)
print("Accuracy:",accuracy)
print("F1 score:",f1)
print("Specificity:",specificity)

print("---------------")


y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label="'1'")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="KNN, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# #NV Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

bayes_predicted = gnb.predict(X_test)

matrix = plot_confusion_matrix(gnb, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our  NV Bayes')
plt.show(matrix)
plt.show()



tp=0
fp=0
tn=0
fn=0

for i in range(len(desired)):
    if desired[i]=="'1'" and bayes_predicted[i]=="'1'":
        tp=tp+1
    elif desired[i]=="'1'" and bayes_predicted[i]=="'-1'":
        fn=fn+1
    elif desired[i]=="'-1'" and bayes_predicted[i]=="'-1'":
        tn=tn+1
    else:
        fp=fp+1
print("Results for NV Bayes:")
print("True Positive:",tp)
print("False Positive:",fp)
print("True Negative:",tn)
print("False Negative:",fn)

precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp + tn + fp + fn)

f1= 2*(precision*recall)/(precision+recall)

specificity = (tn)/(tn+fp)


print("Precision:",precision)
print("Recall:",recall)
print("Accuracy:",accuracy)
print("F1 score:",f1)
print("Specificity:",specificity)

print("---------------")


y_pred_proba = gnb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label="'1'")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="NV Bayes, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# #SVM
from sklearn.svm import SVC
svm = SVC(probability=True)
svm.fit(X_train, y_train)

svm_predicted = svm.predict(X_test)


matrix = plot_confusion_matrix(svm, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for SVM')
plt.show(matrix)
plt.show()

tp=0
fp=0
tn=0
fn=0

for i in range(len(desired)):
    if desired[i]=="'1'" and svm_predicted[i]=="'1'":
        tp=tp+1
    elif desired[i]=="'1'" and svm_predicted[i]=="'-1'":
        fn=fn+1
    elif desired[i]=="'-1'" and svm_predicted[i]=="'-1'":
        tn=tn+1
    else:
        fp=fp+1
print("Results for SVM:")
print("True Positive:",tp)
print("False Positive:",fp)
print("True Negative:",tn)
print("False Negative:",fn)
precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp + tn + fp + fn)

f1= 2*(precision*recall)/(precision+recall)

specificity = (tn)/(tn+fp)


print("Precision:",precision)
print("Recall:",recall)
print("Accuracy:",accuracy)
print("F1 score:",f1)
print("Specificity:",specificity)

print("---------------")





y_pred_proba = svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label="'1'")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="SVM, auc="+str(auc))
plt.legend(loc=4)
plt.show()