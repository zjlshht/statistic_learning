import pandas as pd


def wash_data(column, default):
    if default != default:
        default=0
    res = []
    for i in column:
        tmp = i if i == i else default
        res.append(tmp)
    return res

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
RPY_before = ['COMPL',
              'DEP',
              'NONCOM',
              'IND',
              'FEMALE',
              'MALE',
              'FIRSTGEN',
              'NOTFIRSTGEN',
              'HI_INC',
              'MD_INC',
              'LO_INC',
              'NOPELL']
RPY_weight = {
    'COMPL': 1,
    'DEP': 0.8,
    "NONCOM": 1.5,
    "IND": 1,
    "FEMALE": 1,
    "MALE": 1,
    "FIRSTGEN": 1.1,
    "NOTFIRSTGEN": 1,
    "HI_INC": 0.5,
    "MD_INC": 1,
    "LO_INC": 2,
    "NOPELL": 1.1
}
RPY_final = ['N', 'RT']
RPY_index_up = [0 for i in range(n)]
RPY_index_dowm = RPY_index_up.copy()
RPY_index_by_type = RPY_index_up.copy()
RPY_index = RPY_index_up.copy()
tmp = []
res = 0
for before in RPY_before:
    res_up = 0
    res_dowm = 0
    for year in RPY_year:
        numbers = before+'_RPY_'+year+'YR_'+RPY_final[0]
        rate = before+'_RPY_'+year+'YR_'+RPY_final[1]
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
DBRR_index_up = [0 for i in range(n)]
DBRR_index_dowm = DBRR_index_up.copy()
DBRR_index_by_type = DBRR_index_up.copy()
DBRR_index = DBRR_index_up.copy()
def judge(name,column):
    if name in column:
        return True
    else:
        return False
for loan in LOAN:
    for group in GROUP:
        for year in YEAR:
            rt_column_name='DBRR'+year+'_'+loan+'_'+group+'_RT'
            n_colunm_name='DBRR'+year+'_'+loan+'_'+group+'_N'
            if judge(rt_column_name,columns):
                rt_column=df[rt_column_name]
                n_column=df[n_colunm_name]
            _rt_column=wash_data(rt_column.values,rt_column.mean())
            _n_column=wash_data(n_column.values,n_column.mean())
            for i in range(n):
                DBRR_index_up+=

