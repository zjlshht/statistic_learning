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
    if name in column:
        return True
    else:
        return False

df = pd.read_csv('MERGED2018_19_PP.csv')
column_name = df.columns
CDR2=get_data_from_df('CDR2')
CDR3=get_data_from_df('CDR3')
CDR_index=CDR2/2+CDR3/3


n = len(CDR2)


#RPY
years = ['1', '3', '5', '7']
student_type = ['',
              'COMPL_',
              'NONCOM_',
              'HI_INC_',
              'MD_INC_',
              'LO_INC_']
type_weight = {
    '': 1,
    'COMPL_': 1,
    "NONCOM_": 1.5,
    "HI_INC_": 0.5,
    "MD_INC_": 1,
    "LO_INC_": 2
}
RPY_ac_type = np.zeros(n)
for Type in type_weight:
    RPY_index_ac_by_N=np.zeros(n)
    RPY_ac_N=np.zeros(n)
    for year in years:
        numbers = Type+'RPY_'+year+'YR_'+'N'
        rate = Type+'RPY_'+year+'YR_'+'RT'
        RPY_N = get_data_from_df(numbers,True)
        RPY_RT = get_data_from_df(rate,True)
        RPY_index_ac_by_N += RPY_N * RPY_RT/int(year)
        RPY_ac_N += RPY_N
    RPY_index_ac_by_type=RPY_index_ac_by_N*type_weight[Type]/(RPY_ac_N+1e-8)
    RPY_ac_type+=type_weight[Type]
RPY_index=RPY_index_ac_by_type/RPY_ac_type

# DBRR

GROUP = ['UG', 'UGCOMP', 'GR', 'GRCOMP']
YEAR = ['1', '4', '5', '10', '20']
DBRR_index = np.zeros(n)

for group in GROUP:
    DBRR_index_up = np.zeros(n)
    DBRR_index_dowm = np.zeros(n)
    for year in YEAR:
        DBRR_rt_name = 'DBRR'+year+'_'+'FED'+'_'+group+'_RT'
        DBRR_n_name = 'DBRR'+year+'_'+'FED'+'_'+group+'_N'
        if judge(DBRR_rt_name, column_name):
            DBRR_rt = get_data_from_df(DBRR_rt_name,True)  
            DBRR_n = get_data_from_df(DBRR_n_name,True)
        else:
            continue
        DBRR_index_up=DBRR_rt*DBRR_n*int(year)
        DBRR_index_down=DBRR_n*int(year)
    DBRR_index_by_group=DBRR_index_up/(DBRR_index_down+1e-8) 
    DBRR_index+=DBRR_index_by_group/4
