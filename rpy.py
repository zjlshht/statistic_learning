from datetime import date
from os import X_OK
import pandas as pd
import numpy as np


def wash_data(column, default):
    if default != default:
        default = 0
    res = column.copy()
    for i in range(len(column)):
        tmp = column[i] if column[i] == column[i] else default
        res[i] = tmp
    return res


def str2int(x):
    try:
        res = float(x)
        if res == res:
            return res
        else:
            return -1
    except:
        return -1


def wash_str_data(column):
    res = column.copy()
    up = 0
    down = 1e-6
    for i in range(len(column)):
        tmp = str2int(column[i])
        res[i] = tmp
        if tmp != -1:
            up += tmp
            down += 1
    avg = up/down
    res[res == -1] = avg
    return res


def judge(name, column):
    if name in column:
        return True
    else:
        return False

def build_x_y():
    # read file
    df = pd.read_csv('MERGED2018_19_PP.csv')
    columns = df.columns


    # CDR_INDEX
    CDR2 = df['CDR2']
    CDR3 = df['CDR3']
    _CDR2 = wash_data(CDR2.values, 0)
    _CDR3 = wash_data(CDR3.values, 0)
    n = len(CDR2)
    CDR_index = _CDR2/2+_CDR3/3


    # RPY_INDEX
    RPY_year = ['1', '3', '5', '7']
    RPY_before = ['',
                'COMPL_',
                'DEP_',
                'NONCOM_',
                'IND_',
                'FEMALE_',
                'MALE_',
                'FIRSTGEN_',
                'NOTFIRSTGEN_',
                'HI_INC_',
                'MD_INC_',
                'LO_INC_',
                'NOPELL_',
                'PELL_']
    RPY_weight = {
        '': 1,
        'COMPL_': 1,
        'DEP_': 0.8,
        "NONCOM_": 1.5,
        "IND_": 1,
        "FEMALE_": 1,
        "MALE_": 1,
        "FIRSTGEN_": 1.1,
        "NOTFIRSTGEN_": 1,
        "HI_INC_": 0.5,
        "MD_INC_": 1,
        "LO_INC_": 2,
        "NOPELL_": 1.1,
        "PELL_":1
    }
    RPY_final = ['N', 'RT']
    RPY_index_up = [0 for i in range(n)]
    RPY_index_dowm = RPY_index_up.copy()
    RPY_index_by_type = RPY_index_up.copy()
    RPY_index = RPY_index_up.copy()
    tmp = []
    res = 0
    for before in RPY_before:
        # 每个前缀
        for year in RPY_year:
            # 每个年
            numbers = before+'RPY_'+year+'YR_'+RPY_final[0]  # 数值string
            rate = before+'RPY_'+year+'YR_'+RPY_final[1]
            RPY_N = df[numbers]
            _RPY_N = wash_data(RPY_N.values, RPY_N.mean())
            RPY_RT = df[rate]
            _RPY_RT = wash_data(RPY_RT.values, RPY_RT.mean())
            for i in range(n):
                RPY_index_up[i] += _RPY_N[i]*_RPY_RT[i]/int(year)
                RPY_index_dowm[i] += _RPY_N[i]
        for i in range(n):
            RPY_index_by_type[i] = RPY_index_up[i]/(RPY_index_dowm[i]+1)
            RPY_index[i] += RPY_index_by_type[i]*RPY_weight[before]


    # DBRR
    LOAN = ['FED', 'PP']
    GROUP = ['UG', 'UGCOMP', 'UGNOCOMP', 'UGUNK', 'GR', 'GRCOMP', 'GRNOCOMP']
    YEAR = ['1', '4', '5', '10', '20']
    METRICS = ['RT', 'N']
    LOAN_weight = {
        "FED": 1,
        "PP": 0.5
    }
    GROUP_weight = {
        "UG": 1,
        "GR": 1.2,
        "UGCOMP": 0.8,
        "UGNOCOMP": 0.5,
        "UGUNK": 0.5,
        "GRCOMP": 1,
        "GRNOCOMP": 0.5
    }
    DBRR_index_by_type = [0 for i in range(n)]
    DBRR_index = [0 for i in range(n)]
    for loan in LOAN:
        #fed or pp
        for group in GROUP:
            # all group
            DBRR_index_up = [0 for i in range(n)]
            DBRR_index_dowm = [0 for i in range(n)]
            for year in YEAR:
                rt_column_name = 'DBRR'+year+'_'+loan+'_'+group+'_RT'
                n_colunm_name = 'DBRR'+year+'_'+loan+'_'+group+'_N'
                if judge(rt_column_name, columns):
                    rt_column_tmp = df[rt_column_name].values  # array-str
                    n_column_tmp = df[n_colunm_name].values
                else:
                    continue
                _rt_column = wash_str_data(rt_column_tmp)
                _n_column = wash_str_data(n_column_tmp)
                for i in range(n):
                    DBRR_index_up[i] += _n_column[i] * \
                        _rt_column[i]*int(year)  # 不同年的分子求和
                    DBRR_index_dowm[i] += _n_column[i]  # 不同年的分母求和
            for i in range(n):
                DBRR_index_by_type[i] = DBRR_index_up[i] / \
                    (DBRR_index_dowm[i]+1e-6)  # 年加权nrt
                DBRR_index[i] += DBRR_index_by_type[i] * \
                    LOAN_weight[loan]*GROUP_weight[group]


    # BBRR
    LOAN = ['FED', 'PP']
    GROUP = ['UG', 'UGCOMP', 'UGNOCOMP', 'UGUNK', 'GR', 'GRCOMP', 'GRNOCOMP']
    YEAR = ['1', '2']
    Status = ['DFLT',
            'DLNQ',
            'FBR',
            'DFR',
            'NOPROG',
            'MAKEPROG',
            'PAIDINFULL',
            'DISCHARGE']
    LOAN_weight = {
        "FED": 1,
        "PP": 0.5
    }
    GROUP_weight = {
        "UG": 1,
        "GR": 1.2,
        "UGCOMP": 0.8,
        "UGNOCOMP": 0.5,
        "UGUNK": 0.5,
        "GRCOMP": 1,
        "GRNOCOMP": 0.5
    }
    Status_weight = {
        "DFLT": 1,
        "DLNQ": 0.8,
        "FBR": 0.5,
        "DFR": 0.2,
        "NOPROG": 0.5,
        "MAKEPROG": 0.2,
        "PAIDINFULL": 0,
        "DISCHARGE": 0
    }
    BBRR_index_by_type = [0 for i in range(n)]
    BBRR_index = [0 for i in range(n)]
    for loan in LOAN:
        for group in GROUP:
            BBRR_index_up = [0 for i in range(n)]
            BBRR_index_dowm = [0 for i in range(n)]
            for year in YEAR:
                BBRR_index_up_tmp = [0 for i in range(n)]
                base_name = 'BBRR'+year+'_'+loan+'_'+group+'_'
                numbers_name = base_name+'N'
                if judge(numbers_name, columns):
                    numbers = df[numbers_name].values
                    _n_column = wash_str_data(numbers)
                else:
                    continue
                for status in Status:
                    column_name = base_name+status
                    column = df[column_name].values
                    _column = wash_str_data(column)
                    for i in range(n):
                        BBRR_index_up_tmp[i] += _column[i] * \
                            Status_weight[status]*_n_column[i]  # status 求和
                for i in range(n):
                    BBRR_index_dowm[i] += _n_column[i]
                    BBRR_index_up[i] += BBRR_index_up_tmp[i]*int(year)
            for i in range(n):
                BBRR_index_by_type[i] = BBRR_index_up[i] / \
                    (BBRR_index_dowm[i]+1e-6)  # 年加权nrt
                BBRR_index[i] += BBRR_index_by_type[i] * \
                    LOAN_weight[loan]*GROUP_weight[group]

    Final_index = np.zeros(n)
    for i in range(n):
        Final_index[i] = DBRR_index[i]*BBRR_index[i] / \
            (BBRR_index[i]+6)/(DBRR_index[i]+20)/(1-CDR_index[i])/(1-RPY_index[i])
    max_index = max(Final_index)
    min_index = min(Final_index)
    strandard_index = np.zeros(n)
    for i in range(n):
        standard_index[i] = (Final_index[i]-min_index)/(max_index-min_index)

    X_data = []
    minority_serving = {'HBCU', 'PBI', 'ANNHI', 'AANAPII', 'HSI', 'NANTI'}
    Ms = np.zeros(n)
    for i in minority_serving:
        tmp = df[i].values
        Ms_type = wash_data(tmp, 0)
        for j in range(n):
            Ms[j] += Ms_type[j]
    X_data.append(Ms)


    ACT_class = ['CM', 'EN', 'MT', 'WR']
    ACT_quan = ['25', 'MID', '75']
    for classs in ACT_class:
        for quan in ACT_quan:
            name = 'ACT'+classs+quan
            data = df[name]
            _data = wash_data(data.values, data.mean())
            X_data.append(_data)

    HIGHDEG = df['HIGHDEG'].values
    X_data.append(HIGHDEG)

    HCM2 = df['HCM2'].values
    X_data.append(HCM2)

    CIP_dict = []
    with open('cip.txt', 'r') as f:
        for i in f.readlines():
            CIP_dict.append(i.strip())

    CIP_index_1 = np.zeros(n)
    CIP_index_2 = np.zeros(n)
    for i in CIP_dict:
        data = df[i]
        _data = wash_data(data.values, 0)
        for j in range(n):
            if j == 0:
                continue
            elif j == 1:
                CIP_index_1[j] += 1
            else:
                CIP_index_2[j] += 1
    X_data.append(CIP_index_2)
    X_data.append(CIP_index_1)
    INEXPFTE_tmp = df['INEXPFTE']
    INEXPFTE = wash_data(INEXPFTE_tmp.values, INEXPFTE_tmp.mean())
    X_data.append(INEXPFTE)

    AVGFACSAL_tmp = df['AVGFACSAL']
    AVGFACSAL = wash_data(AVGFACSAL_tmp.values, AVGFACSAL_tmp.mean())
    X_data.append(AVGFACSAL)

    C_base = ['C100_4_POOLED', 'C100_L4_POOLED', 'C200_4_POOLED',
            'C200_L4_POOLED', 'C150_4_POOLED', 'C150_L4_POOLED']
    for i in C_base:
        data = df[i]
        _data = wash_data(data.values, data.mean())
        X_data.append(_data)

    CONTROL_1=np.zeros(n)
    CONTROL_2=np.zeros(n)
    CONTROL_3=np.zeros(n)
    CONTROL=df['CONTROL'].values
    for i in range(n):
        if i==1:
            CONTROL_1[i]=1
        elif i==2:
            CONTROL_2[i]=1
        else:
            CONTROL_3[i]=1
    X_data.append(CONTROL_1)
    X_data.append(CONTROL_2)
    X_data.append(CONTROL_3)

    Ratio_indexs=['FTFTPCTPELL','FTFTPCTFLOAN']
    for i in Ratio_indexs:
        data=df[i]
        _data=wash_data(data.values,data.mean())
        X_data.append(_data)

    MAIN=df['MAIN'].values
    X_data.append(MAIN)

    LATITUDE_tmp=df['LATITUDE']
    LATITUDE=wash_data(LATITUDE_tmp.values,0)
    LATITUDE/=90
    X_data.append(LATITUDE)

    LONGITUDE_tmp=df['LONGITUDE']
    LONGITUDE=wash_data(LONGITUDE_tmp.values,0)
    LONGITUDE/=180
    X_data.append(LONGITUDE)

    COSTT4_P_tmp=df['COSTT4_P']
    COSTT4_P=wash_data(COSTT4_P_tmp.values,COSTT4_P_tmp.mean())
    X_data.append(COSTT4_P)

    COSTT4_A_tmp=df['COSTT4_A']
    COSTT4_A=wash_data(COSTT4_A_tmp.values,COSTT4_A_tmp.mean())
    X_data.append(COSTT4_A)

    Student_class = ['',
                'DEP_',
                'NOLOAN_',
                'IND_',
                'FEMALE_',
                'MALE_',
                'FIRSTGEN_',
                'NOT1STGEN_',
                'HI_INC_',
                'MD_INC_',
                'LO_INC_',
                'NOPELL_',
                'PELL_']
    Student_status = [
        'COMP',
        'WDRAW',
        'ENRL',
        'UNKN'
    ]
    YEAR=['2','3','4','6','8']
    WDRAW_index=np.zeros(n)
    COMP_index=np.zeros(n)
    for classs in Student_class:
        for year in YEAR:
            name1= classs+'WDRAW'+'_2YR_'+'TRANS_YR'+year+'_RT'
            name2= classs+'WDRAW'+'_4YR_'+'TRANS_YR'+year+'_RT'
            name3= classs+'COMP'+'_2YR_'+'TRANS_YR'+year+'_RT'
            name4= classs+'COMP'+'_4YR_'+'TRANS_YR'+year+'_RT'
            column1=df[name1]
            _column1=wash_data(column1.values,column1.mean())
            column2=df[name2]
            _column2=wash_data(column2.values,column2.mean())
            column3=df[name3]
            _column3=wash_data(column3.values,column3.mean())
            column4=df[name4]
            _column4=wash_data(column4.values,column4.mean())
            WDRAW_index+=_column1
            WDRAW_index+=_column2
            COMP_index+=_column3
            COMP_index+=_column4
    X_data.append(WDRAW_index)
    X_data.append(COMP_index)


    MTHCMP=np.zeros(n)
    MTHCMP_type=['1','2','3','4','5','6']
    for types in MTHCMP_type:
        name='MTHCMP'+types
        column=df[name]
        _column=wash_data(column.values,column.mean())
        MTHCMP+=_column
    X_data.append(MTHCMP)


    OPEFLAG=df['OPEFLAG'].values
    OPEFLAG[OPEFLAG!=1]=0
    X_data.append(OPEFLAG)

    Tuition_type=['41','42','43','44','45','4_048','4_3075','4_75UP','4']
    School_type=['OTHER','PRIV','PROG','PUB']
    for tuition in Tuition_type:
        for _type in School_type:
            name = 'NPT'+tuition+'_'+_type
            column=df[name]
            _column=wash_data(column.values,0)
            X_data.append(_column)

    RET_year=['4','L4']
    RET_type=['FT','PT']
    for year in RET_year:
        for types in RET_type:
            name = 'RET'+'_'+types+year
            column=df[name]
            _column=wash_data(column.values,column.mean())
            X_data.append(_column)

    X_data_final=np.array(X_data)
    X=X_data_final.T
    return [X,standard_index]

#lr



