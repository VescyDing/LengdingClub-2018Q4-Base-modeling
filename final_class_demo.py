# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:15:16 2019

@author: DWX
"""

import pandas as pd
import numpy as np
import sys

df = pd.read_csv("./LoanStats3a 2018Q4.csv", skiprows = 1, low_memory = True)
print(df.head(10))
print(df.info())


df.drop('id', axis = 1, inplace = True)
df.drop('member_id', axis = 1, inplace = True)

df.term.replace(to_replace = '[^0-9]+', value = '', inplace = True)
df.int_rate.replace('%', value = '', inplace = True, regex = True) # 这次替换掉了,神奇
df.revol_util.replace('%', value = '', inplace = True, regex = True) # 这次替换掉了,神奇


print(df.int_rate.value_counts())

df.drop('sub_grade', axis = 1, inplace = True)
df.drop('emp_title', axis = 1, inplace = True)

df.emp_length.replace('n/a', np.NaN, inplace = True, regex = True)
df.emp_length.replace(to_replace = '[^0-9]+',value = '', inplace = True, regex = True)

df.dropna(axis = 1, how = 'all', inplace = True)
df.dropna(axis = 0, how = 'all', inplace = True)

print(df.info())

# =============================================================================
# debt_settlement_flag_date     160 non-null object
# settlement_status             160 non-null object
# settlement_date               160 non-null object
# settlement_amount             160 non-null float64
# settlement_percentage         160 non-null float64
# settlement_term               160 non-null float64
# =============================================================================


df.drop(['debt_settlement_flag_date', 'settlement_status', 'settlement_date',\
         'settlement_amount', 'settlement_percentage', 'settlement_term', 'total_acc', 'loan_amnt'],\
axis = 1, inplace = True)

print(df.info())

# =============================================================================
# col collections_12_mths_ex_med has 2
# col policy_code has 1
# col acc_now_delinq has 3
# col chargeoff_within_12_mths has 2
# col delinq_amnt has 4
# col pub_rec_bankruptcies has 4
# col tax_liens has 3
# col delinq_2yrs has 13
# col inq_last_6mths has 29
# col mths_since_last_delinq has 96
# col mths_since_last_record has 114
# col open_acc has 45
# col pub_rec has 7
# col out_prncp has 1
# col out_prncp_inv has 1
# =============================================================================

df.drop(['collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', 'chargeoff_within_12_mths',\
         'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens', 'delinq_2yrs', 'inq_last_6mths',\
         'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'out_prncp', 'out_prncp_inv'],\
axis = 1, inplace = True)

for col in df.select_dtypes(include = ['float']).columns:
    print('col {} has {}'.format(col, len(df[col].unique())))
    
# =============================================================================
# col term has 2
# col int_rate has 394
# col grade has 7
# col emp_length has 11
# col home_ownership has 5
# col verification_status has 3
# col issue_d has 55
# col loan_status has 4 这个是标签,不能删
# col pymnt_plan has 1
# col purpose has 14
# col zip_code has 837
# col addr_state has 50
# col earliest_cr_line has 531
# col initial_list_status has 1
# col last_pymnt_d has 113
# col next_pymnt_d has 99
# col last_credit_pull_d has 141
# col application_type has 1
# col hardship_flag has 1
# col disbursement_method has 1
# col debt_settlement_flag has 2
# =============================================================================

print('='*30)

df.drop(['debt_settlement_flag', 'disbursement_method', 'hardship_flag', 'application_type',\
         'last_credit_pull_d', 'next_pymnt_d', 'last_pymnt_d', 'initial_list_status', 'earliest_cr_line',\
         'addr_state', 'zip_code', 'purpose', 'pymnt_plan', 'issue_d', 'verification_status',\
         'home_ownership', 'emp_length', 'grade', 'term', 'desc', 'title'],\
axis = 1, inplace = True)


for col in df.select_dtypes(include = ['object']).columns:
    print('col {} has {}'.format(col, len(df[col].unique())))


#df.to_csv('./df_data1.csv') 持久化

df.loan_status.replace('Fully Paid', value=int(1), inplace = True)
df.loan_status.replace('Charged Off', value=int(0), inplace = True) 
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid', value=np.NaN, inplace = True) 
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off', value=np.NaN, inplace = True) 

df.dropna(axis = 0, how = 'any', inplace = True)
df.fillna(0.0, inplace = True)

print(df.info())

cor =df.corr() # 求协方差矩阵
cor.iloc[:, :] = np.tril(cor, k=-1) 

# =============================================================================
# np.tril: numpy.tril（m，k = 0 ）[来源]
# 数组的下三角形。
# 
# 返回数组的副本，其中元素位于第k个对角线上方。
# 
# 参数：	
# m ： array_like，shape（M，N）
# 输入数组。
# 
# k ： int，可选
# 对角线以上为零元素。 k = 0（默认值）是主对角线，k <0低于它，k> 0高于它。
# pandas.iloc: 矩阵的切片, [:, :]即所有行所有列
# =============================================================================

cor = cor.stack() # stack是一个排序函数, 可以让数据的表格结构的行列索引变成只有列索引

print(cor[(cor>0.55)|(cor<-0.55)])

# =============================================================================
# funded_amnt_inv          funded_amnt        0.958601
# installment              funded_amnt        0.956064
#                          funded_amnt_inv    0.905080
# total_pymnt              funded_amnt        0.901822
#                          funded_amnt_inv    0.880515
#                          installment        0.852602
# total_pymnt_inv          funded_amnt        0.869617
#                          funded_amnt_inv    0.911760
#                          installment        0.813329
#                          total_pymnt        0.971582
# total_rec_prncp          funded_amnt        0.872058
#                          funded_amnt_inv    0.848349
#                          installment        0.848884
#                          total_pymnt        0.972124
#                          total_pymnt_inv    0.941108
# total_rec_int            funded_amnt        0.736702
#                          funded_amnt_inv    0.730422
#                          installment        0.633291
#                          total_pymnt        0.836580
#                          total_pymnt_inv    0.823605
#                          total_rec_prncp    0.696938
# collection_recovery_fee  recoveries         0.808311
# =============================================================================


df.drop(['total_pymnt', 'funded_amnt'], axis= 1, inplace = True)

#df.to_csv('./df_data2.csv')

print(df.info())

# 数据清洗就此结束