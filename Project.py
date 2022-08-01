#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[96]:


data = pd.read_csv("data.csv",delimiter=',')
data.head()


# In[97]:


#Нормализуем данные и раделим на тренировочный, валидационный и тестовый наборы в соотношении 60:20:20
for i in ['MIP', 'STDIP', 'EKIP', 'SIP', 'MC', 'STDC', 'EKC', 'SC']:
    data[i] = (data[i] - data[i].min())/(data[i].max() - data[i].min())
train= data.head(int(len(data)*0.6))
valid=data.iloc[10738:14319]
test=data.tail(int(len(data)*0.2))
#Запишем значения в файлы
train.to_csv("train.csv")
valid.to_csv("valid.csv")
test.to_csv("test.csv")


# In[98]:


train.tail()


# In[99]:


test.tail()


# In[100]:


valid.head()


# In[101]:


valid.tail()


# In[102]:


X = pd.DataFrame(train.drop(['TARGET'], axis=1))


# In[103]:


Y = pd.DataFrame(train['TARGET'])


# In[104]:


X_test = pd.DataFrame(test.drop(['TARGET'], axis=1))


# In[105]:


Y_test = pd.DataFrame(test['TARGET'])


# In[106]:


#Линейная регрессия
reg = LinearRegression().fit(X, Y)


# In[107]:


t=reg.predict(X_test)


# In[108]:


t1=Y_test.to_numpy() #t1- массив с тестовым откликом


# In[109]:


#Переведем полученные значения в 0 и 1 . Если значение >=0.5, то 1, иначе 0
for a in range (len(t)):
    if t[a]>=0.5 :
        t[a]=1
    else:
        t[a]=0


# In[110]:


np.savetxt('result1', t)
#t.to_csv("res_lin_reg.csv")


# In[111]:


n=0
for a in range (len(t1)):
    if t[a]==t1[a] :
        n+=1
print(n)


# In[112]:


r2 = reg.score(X,Y)
r2


# In[113]:


f1_score(Y_test, t, average='macro')


# In[28]:


#Из 3579 значений алгоритм распознал 3542 


# In[114]:


importance = reg.coef_  #Значение коэффициентов  θ1,…,θp :
importance


# In[115]:


reg.intercept_ #Значение коэффициента  θ0 :


# In[116]:


X1 = pd.DataFrame(train.drop(['TARGET'], axis=1))


# In[117]:


y1 = pd.DataFrame(train['TARGET']).values.ravel()


# In[118]:


#kNN-метод 
neigh = KNeighborsClassifier(n_neighbors=21, p=2)
neigh.fit(X1, y1)


# In[119]:


X_valid = pd.DataFrame(valid.drop(['TARGET'], axis=1))
Y_valid = pd.DataFrame(valid['TARGET'])


# In[120]:


res_knn=neigh.predict(X_valid)


# In[121]:


res_knn #массив с результатами knn для валидационного набора


# In[122]:


neigh.kneighbors(X_valid)


# In[123]:


t2=Y_valid.to_numpy()  #t2 - массив с валидационными данными отклика
k=0
for a in range (len(res_knn)):
    if res_knn[a]==t2[a] :
        k+=1
print(k)


# In[124]:


res_knn_test=neigh.predict(X_test)


# In[125]:


neigh.kneighbors(X_test)


# In[126]:


#t3=Y_test.to_numpy()
q=0
for a in range (len(res_knn_test)):
    if res_knn_test[a]==t1[a] :
        q+=1
print(q)


# In[127]:


r2 = neigh.score(X_test,Y_test)
r2


# In[128]:


f1_score(Y_test, res_knn_test, average='macro')


# In[ ]:


#Логистическая регрессия


# In[65]:


#обучение модели
reg_log = LogisticRegression(penalty='l1', random_state=2019, solver='saga').fit(X, Y.values.ravel())


# In[66]:


res_log_valid=reg_log.predict(X_valid)


# In[67]:


res_log_valid

b=0
for a in range (len(res_log_valid)):
    if res_log_valid[a]==t2[a] :
        b+=1
print(b)
r2 = reg_log.score(X_valid,Y_valid)
r2


# In[68]:


res_log_test=reg_log.predict(X_test)


# In[69]:


t3=Y_test.to_numpy()

b=0
for a in range (len(res_log_test)):
    if res_log_test[a]==t3[a] :
        b+=1
print(b)
r2 = reg_log.score(X_test,Y_test)
r2


# In[70]:


#отбор отклика Y из тестовых данных и преобразование в массив
Y_true = (test['TARGET'].to_frame().T).values.ravel()


# In[71]:


Y_pred_probs = reg_log.predict_proba(X_test)


# In[72]:


#отбор вероятностей отнесения объектов к классу 1
Y_pred_probs_class_1 = Y_pred_probs[:, 1]


# In[73]:


#подключение библиотеки для вычисления метрик
from sklearn import metrics


# In[74]:


fpr, tpr, _ = metrics.roc_curve(Y_true, res_log_test)


# In[75]:


#вычисляем AUC
metrics.roc_auc_score(Y_true, Y_pred_probs_class_1)


# In[76]:


#вычисление Recall
metrics.recall_score(Y_true, res_log_test)


# In[77]:


#вычисление Precision
metrics.precision_score(Y_true, res_log_test)


# In[93]:


f1_score(Y_test, res_log_test, average='macro')


# In[80]:


metrics.plot_roc_curve(reg_log, X_test, Y_true, color='darkorange') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()


# In[83]:


from matplotlib.pylab import rc, plot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[86]:


report = classification_report(Y_test, reg_log.predict(X_test), target_names=['Non-churned', 'Churned'])
print(report)


# In[84]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

font = {'size' : 15}

plt.rc('font', **font)

cnf_matrix = confusion_matrix(Y_test, reg_log.predict(X_test))
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=['Non-churned', 'Churned'],
                      title='Confusion matrix')
plt.savefig("conf_matrix.png")
plt.show()


# In[1]:


from sklearn.tree import DecisionTreeClassifier


# In[167]:


features = list(train.columns[:8])
x = train[features]
y = train['TARGET']


# In[168]:


#Дерево решений
tree = DecisionTreeClassifier(criterion='entropy', #критерий разделения
                              min_samples_leaf=10, #минимальное число объектов в листе
                              max_leaf_nodes=30, #максимальное число листьев
                              random_state=2020)
clf=tree.fit(x, y)


# In[169]:


features = list(valid.columns[:8])
x1 = valid[features]
y_true = valid['TARGET']
y_pred = clf.predict(x1)


# In[170]:


features = list(test.columns[:8])
x2 = test[features]
y_true = test['TARGET']
y_pred = clf.predict(x2)


# In[171]:


k=0
for a in range (len(y_pred)):
    if y_pred[a]==t3[a] :
        k+=1
print(k)


# In[174]:


from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)


# In[175]:


from sklearn.metrics import f1_score
f1_score(y_true, y_pred) #average='macro')


# In[176]:


f1_score(y_true, y_pred, average='macro')


# In[177]:


clf.tree_.max_depth


# In[166]:


r2 = clf.score(X_valid,Y_valid)
r2


# In[64]:


report = classification_report(y_true, clf.predict(x1), target_names=['Non-churned', 'Churned'])
print(report)


# In[87]:


from matplotlib.pylab import rc, plot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split


# In[88]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

font = {'size' : 15}

plt.rc('font', **font)

cnf_matrix = confusion_matrix(Y_test, clf.predict(X_test))
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=['Non-churned', 'Churned'],
                      title='Confusion matrix')
plt.savefig("conf_matrix.png")
plt.show()


# In[284]:


from sklearn.neural_network import MLPClassifier


# In[291]:


MLP = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (8,2),max_iter=10000, alpha = 1e-5, random_state = 1)
MLP.fit(X, Y)


# In[286]:


Y_mlpv=MLP.predict(X_valid)


# In[287]:


Y_mlpv


# In[288]:


b=0
for a in range (len(Y_mlpv)):
    if Y_mlpv[a]==t2[a] :
        b+=1
print(b)


# In[289]:


r2 = MLP.score(X_valid,Y_valid)
r2


# In[290]:


f1_score(Y_valid, Y_mlpv, average='macro')


# In[270]:


accuracy_score(Y_valid, Y_mlpv)


# In[295]:


Y_mlp_test=MLP.predict(X_test)


# In[296]:


b=0
for a in range (len(Y_mlp_test)):
    if Y_mlp_test[a]==t3[a] :
        b+=1
print(b)


# In[298]:


accuracy_score(Y_test, Y_mlp_test)


# In[299]:


f1_score(Y_test, Y_mlp_test, average='macro')


# In[ ]:




