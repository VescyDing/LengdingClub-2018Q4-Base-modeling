# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:49:57 2019

@author: DWX
"""

import pandas as pd
from sklearn.externals import joblib # 模型持久化
from sklearn.model_selection import train_test_split # 数据集分割
from sklearn.linear_model.logistic import LogisticRegression #逻辑回归
from sklearn import metrics #有监督学习的评判标准
from sklearn.ensemble.forest import RandomForestClassifier #随机森林
from sklearn.ensemble import GradientBoostingClassifier #GBTD
from sklearn import svm
import time
from sklearn.model_selection import GridSearchCV #网格搜寻获得超参
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./df_data2.csv')

print(df.loan_status.value_counts())

Y = df.loan_status
X = df.drop(['loan_status', 'recoveries', 'total_rec_prncp', 'collection_recovery_fee'], axis = 1, inplace = False)
#X = df.drop('loan_status', axis = 1, inplace = False)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

lr = LogisticRegression()





start = time.time()
lr.fit(x_train, y_train)
train_predict = lr.predict(x_train)
train_f1 = metrics.f1_score(train_predict, y_train) # 训练集上的准确率
train_acc = metrics.accuracy_score(train_predict, y_train) # 精确率
train_rec = metrics.recall_score(train_predict, y_train) # 召回率
print('逻辑回归效果如下: ')
print('训练集上的f1_mean值为%.4f' % train_f1, end=' ')
print('训练集上的精确率为%.4f' % train_acc, end=' ')
print('训练集上的召回率为%.4f' % train_rec)

test_predict = lr.predict(x_test)
test_f1 = metrics.f1_score(test_predict, y_test) # 测试集上的准确率
test_acc = metrics.accuracy_score(test_predict, y_test) # 精确率
test_rec = metrics.recall_score(test_predict, y_test) # 召回率
print('测试集上的f1_mean值为%.4f' % test_f1, end=' ')
print('测试集上的精确率为%.4f' % test_acc, end=' ')
print('测试集上的召回率为%.4f' % test_rec)

end = time.time()
print(end - start)


print('=' * 30)
rf = RandomForestClassifier()
start = time.time()
rf.fit(x_train, y_train)
train_predict = rf.predict(x_train)
train_f1 = metrics.f1_score(train_predict, y_train) # 训练集上的准确率
train_acc = metrics.accuracy_score(train_predict, y_train) # 精确率
train_rec = metrics.recall_score(train_predict, y_train) # 召回率
print('随机森林效果如下: ')
print('训练集上的f1_mean值为%.4f' % train_f1, end=' ')
print('训练集上的精确率为%.4f' % train_acc, end=' ')
print('训练集上的召回率为%.4f' % train_rec)

test_predict = rf.predict(x_test)
test_f1 = metrics.f1_score(test_predict, y_test) # 测试集上的准确率
test_acc = metrics.accuracy_score(test_predict, y_test) # 精确率
test_rec = metrics.recall_score(test_predict, y_test) # 召回率
print('测试集上的f1_mean值为%.4f' % test_f1, end=' ')
print('测试集上的精确率为%.4f' % test_acc, end=' ')
print('测试集上的召回率为%.4f' % test_rec)

end = time.time()
print(end - start)

# GBTD和SVM太慢,先不跑

#joblib.dump(rf, './model_RandomForestClassifier.m')

feature_importance = rf.feature_importances_ 

feature_importance = 100.0*(feature_importance/feature_importance.max())
index = np.argsort(feature_importance)[-10:] # argsort是倒序
plt.barh(np.arange(10), feature_importance[index], color = 'dodgerblue', alpha = 0.4)
print(np.array(X.columns)[index])
plt.yticks(np.arange(10+0.25), np.array(X.columns)[index])
plt.xlabel('Relative importance')
plt.title('Top 10 Importance Variable')
plt.show()

# =============================================================================
# 其实这个模型的鲁棒性非常的好,即使删掉前三的 recoveries, collection_recovery_fee, 
# total_rec_prncp, 模型的精确率依然能达到98%,Relative importance依然是平滑的下降,并没有
# 因为确实特征和坍塌,仍然具有很强的预测能力!
# =============================================================================




















