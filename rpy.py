import pandas as pd


def wash_data(column, default):
    if default != default:
        default=0
    res = column.copy()
    for i in range(len(column)):
        tmp = column[i] if column[i] == column[i] else default
        res[i] =tmp
    return res


def str2int(x):
    try:
        res = float(x)
        if res==res:
            return res
        else:
            return -1
    except Exception as e:
        return -1


def wash_str_data(column):
    res = column.copy()
    up=0
    down=1e-6
    index=[]
    for i in range(len(column)):
        tmp=str2int(column[i])
        res[i]=tmp
        if tmp != -1:
            up+=tmp
            down+=1
    avg=up/down
    res[res==-1]=avg
    return res


def judge(name,column):
    if name in column:
        return True
    else:
        return False


#read file
df = pd.read_csv('MERGED2018_19_PP.csv')
columns=df.columns


#CDR_INDEX
CDR2 = df['CDR2']
CDR3 = df['CDR3']
_CDR2 =wash_data(CDR2.values,0)
_CDR3 =wash_data(CDR3.values,0)
n = len(CDR2)
CDR_INDEX=_CDR2/2+_CDR3/3


#RPY_INDEX
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
              'NOPELL_']
RPY_weight = {
    '':1,
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
    "NOPELL_": 1.1
}
RPY_final = ['N', 'RT']
RPY_index_up = [0 for i in range(n)]
RPY_index_dowm = RPY_index_up.copy()
RPY_index_by_type = RPY_index_up.copy()
RPY_index = RPY_index_up.copy()
tmp = []
res = 0
for before in RPY_before:
    #每个前缀
    for year in RPY_year:
        #每个年
        numbers = before+'RPY_'+year+'YR_'+RPY_final[0] #数值string
        rate = before+'RPY_'+year+'YR_'+RPY_final[1] #
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


#DBRR
LOAN=['FED','PP']
GROUP=['UG','UGCOMP','UGNOCOMP','UGUNK','GR','GRCOMP','GRNOCOMP']
YEAR=['1','4','5','10','20']
METRICS=['RT','N']
LOAN_weight={
    "FED":1,
    "PP":0.5
}
GROUP_weight={
    "UG":1,
    "GR":1.2,
    "UGCOMP":0.8,
    "UGNOCOMP":0.5,
    "UGUNK":0.5,
    "GRCOMP":1,
    "GRNOCOMP":0.5
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
            rt_column_name='DBRR'+year+'_'+loan+'_'+group+'_RT'
            n_colunm_name='DBRR'+year+'_'+loan+'_'+group+'_N'
            if judge(rt_column_name,columns):
                rt_column_tmp=df[rt_column_name].values#array-str
                n_column_tmp=df[n_colunm_name].values
            else:
                continue
            _rt_column=wash_str_data(rt_column_tmp)#
            _n_column=wash_str_data(n_column_tmp)
            for i in range(n):
                DBRR_index_up[i]+=_n_column[i]*_rt_column[i]*int(year) #不同年的分子求和
                DBRR_index_dowm[i]+=_n_column[i] #不同年的分母求和
        for i in range(n):
            DBRR_index_by_type[i]=DBRR_index_up[i]/(DBRR_index_dowm[i]+1e-6) #年加权nrt    
            DBRR_index[i]+=DBRR_index_by_type[i]*LOAN_weight[loan]*GROUP_weight[group]


#BBRR
LOAN=['FED','PP']
GROUP=['UG','UGCOMP','UGNOCOMP','UGUNK','GR','GRCOMP','GRNOCOMP']
YEAR=['1','2']
Status=['DFLT',
        'DLNQ',
        'FBR',
        'DFR',
        'NOPROG',
        'MAKEPROG',
        'PAIDINFULL',
        'DISCHARGE']
LOAN_weight={
    "FED":1,
    "PP":0.5
}
GROUP_weight={
    "UG":1,
    "GR":1.2,
    "UGCOMP":0.8,
    "UGNOCOMP":0.5,
    "UGUNK":0.5,
    "GRCOMP":1,
    "GRNOCOMP":0.5
}
Status_weight={
    "DFLT":1,
    "DLNQ":0.8,
    "FBR":0.5,
    "DFR":0.2,
    "NOPROG":0.5,
    "MAKEPROG":0.2,
    "PAIDINFULL":0,
    "DISCHARGE":0
}
BBRR_index_by_type = [0 for i in range(n)]
BBRR_index = [0 for i in range(n)]
for loan in LOAN:
    for group in GROUP:
        BBRR_index_up = [0 for i in range(n)]
        BBRR_index_dowm = [0 for i in range(n)]
        for year in YEAR:
            BBRR_index_up_tmp = [0 for i in range(n)]
            base_name='BBRR'+year+'_'+loan+'_'+group+'_'
            numbers_name=base_name+'N'
            if judge(numbers_name,columns):
                numbers=df[numbers_name].values
                _n_column=wash_str_data(numbers)
            else:
                continue
            for status in Status:
                column_name=base_name+status
                column=df[column_name].values
                _column=wash_str_data(column)
                for i in range(n):
                    BBRR_index_up_tmp[i]+=_column[i]*Status_weight[status]*_n_column[i]#status 求和
            for i in range(n):
                BBRR_index_dowm[i]+=_n_column[i]
                BBRR_index_up[i]+=BBRR_index_up_tmp[i]*int(year)
        for i in range(n):
            BBRR_index_by_type[i]=BBRR_index_up[i]/(BBRR_index_dowm[i]+1e-6) #年加权nrt    
            BBRR_index[i]+=BBRR_index_by_type[i]*LOAN_weight[loan]*GROUP_weight[group]             



            
            


