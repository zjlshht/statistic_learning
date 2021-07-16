import pandas as pd
import numpy as np


def str2float(x):
    try:
        res = float(x)
        if res == res:
            return res
        else:
            return -1
    except:
        return -1


def clear_nan(column, replace=False):
    n = len(column)
    result = np.zeros(n)
    for i in range(n):
        result[i] = str2float(column[i])
    if replace:
        num=1e-11
        count=0
        for i in result:
            if i!=-1:
                count+=i
                num+=1
        mean=count/num
        result[result==-1]=mean
    else:
        result[result==-1]=0
    return result


def get_data_from_df(name,replace=False):
    column=df[name].values
    res=clear_nan(column,replace)
    return res


def column_exist(name, column_name):
    if name in column_name:
        return True
    else:
        return False

df = pd.read_csv('MERGED2018_19_PP.csv')
column_name = df.columns
CDR2=get_data_from_df('CDR2')
CDR3=get_data_from_df('CDR3')
CDR_index=CDR2/2+CDR3/2


n = len(CDR2)


#RPY
years = ['1', '3', '5', '7']
student_type = ['',
              'COMPL_',
              'HI_INC_',
              'MD_INC_',
              'LO_INC_']
type_weight = {
    '': 1,
    'COMPL_': 1,
    "HI_INC_": 0.5,
    "MD_INC_": 1,
    "LO_INC_": 2
}
RPY_ac_type = np.zeros(n)
RPY_index_ac_by_type=np.zeros(n)
for Type in type_weight:
    RPY_index_ac_by_N=np.zeros(n)
    RPY_ac_N=np.zeros(n)
    for year in years:
        numbers = Type+'RPY_'+year+'YR_'+'N'
        rate = Type+'RPY_'+year+'YR_'+'RT'
        RPY_N = get_data_from_df(numbers,True)
        RPY_RT = get_data_from_df(rate,True)
        RPY_index_ac_by_N += RPY_N * RPY_RT/int(year)
        RPY_ac_N += RPY_N/int(year)
    RPY_index_ac_by_type+=RPY_index_ac_by_N*type_weight[Type]/(RPY_ac_N+1e-8)
    RPY_ac_type+=type_weight[Type]
RPY_index=RPY_index_ac_by_type/RPY_ac_type

# DBRR

GROUP = ['UG', 'UGCOMP', 'GR', 'GRCOMP']
YEAR = ['1', '4', '5', '10', '20']
DBRR_index = np.zeros(n)

for group in GROUP:
    DBRR_index_up = np.zeros(n)
    DBRR_index_down = np.zeros(n)
    for year in YEAR:
        DBRR_rt_name = 'DBRR'+year+'_'+'FED'+'_'+group+'_RT'
        DBRR_n_name = 'DBRR'+year+'_'+'FED'+'_'+group+'_N'
        if column_exist(DBRR_rt_name, column_name):
            DBRR_rt = get_data_from_df(DBRR_rt_name,True)  
            DBRR_n = get_data_from_df(DBRR_n_name,True)
        else:
            continue
        DBRR_index_up+=DBRR_rt*DBRR_n*int(year)
        DBRR_index_down+=DBRR_n*int(year)
    DBRR_index_by_group=DBRR_index_up/(DBRR_index_down+1e-8) 
    DBRR_index+=DBRR_index_by_group/4

# BBRR
GROUP = ['UG', 'UGCOMP', 'GR', 'GRCOMP']
YEAR = ['1', '2']
Status = ['DFLT',
          'DLNQ',
          'FBR',
          'DFR',
          'NOPROG',
          'MAKEPROG',
          'PAIDINFULL',
          'DISCHARGE']
Status_weight = {
    "DFLT": 1,
    "DLNQ": 0.8,
    "FBR": 0.4,
    "DFR": 0.2,
    "NOPROG": 0.4,
    "MAKEPROG": 0.2,
    "PAIDINFULL": 0,
    "DISCHARGE": 0
}
BBRR_index = np.zeros(n)
for group in GROUP:
    BBRR_index_up = np.zeros(n)
    BBRR_index_down = np.zeros(n)
    for year in YEAR:
        BBRR_index_by_status = np.zeros(n)
        BBRR_name = 'BBRR'+year+'_'+'FED'+'_'+group+'_'
        BBRR_n_name = BBRR_name+'N'
        if column_exist(BBRR_n_name, column_name):
            BBRR_n=get_data_from_df(BBRR_n_name,True)
        else:
            continue
        for status in Status:
            BBRR_rt_name = BBRR_name+status
            BBRR_rt = get_data_from_df(BBRR_rt_name,True)
            BBRR_index_by_status+=BBRR_rt*Status_weight[status]/3
        BBRR_index_down+=BBRR_n*int(year)
        BBRR_index_up+=BBRR_index_by_status*BBRR_n*int(year)
    BBRR_index+=BBRR_index_up/(BBRR_index_down+1e-8)/4

#DEBT
student_type = ['',
              'GRAD_',
              'HI_INC_',
              'MD_INC_',
              'LO_INC_']
type_weight = {
    '': 1,
    'GRAD_': 2,
    "HI_INC_": 1.5,
    "MD_INC_": 1,
    "LO_INC_": 0.5
}
DEBT_index_down=np.zeros(n)
DEBT_index_up=np.zeros(n)
for type_name in student_type:
    DEBT_n_name=type_name+'DEBT_N'
    DEBT_mdn_name=type_name+'DEBT_MDN'
    DEBT_n=get_data_from_df(DEBT_n_name,True)
    DEBT_mdn=get_data_from_df(DEBT_mdn_name,True)
    DEBT_index_up+=DEBT_mdn*DEBT_n*type_weight[type_name]
    DEBT_index_down+=DEBT_n*type_weight[type_name]
DEBT_index=DEBT_index_up/DEBT_index_down

def corr(x,y):
    up=sum(x*y)
    down=np.sqrt(sum(x**2)*sum(y**2))
    return up/down

import matplotlib.pyplot as plt
fig = plt.figure()  
ax = plt.subplot()
ax.boxplot(BBRR_index)
plt.title("BBRR_index")


fig = plt.figure()  
ax = plt.subplot()
ax.boxplot(DBRR_index)
plt.title("DBRR_index")

fig = plt.figure()  
ax = plt.subplot()
ax.boxplot(DEBT_index)
plt.title("DEBT_index")

fig = plt.figure()  
ax = plt.subplot()
ax.boxplot(CDR_index)
plt.title("CDR_index")

fig = plt.figure()  
ax = plt.subplot()
ax.boxplot(RPY_index)
plt.title("RPY_index")