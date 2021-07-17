import pandas as pd
import numpy as np


#函数名字可以改
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


# read file  读的文件可以改
df = pd.read_csv('MERGED2018_19_PP.csv')
columns = df.columns


# CDR_INDEX
CDR2 = df['CDR2']
CDR3 = df['CDR3']
_CDR2 = wash_data(CDR2.values, 0)
_CDR3 = wash_data(CDR3.values, 0)
n = len(CDR2)
CDR_index = _CDR2/2+_CDR3/3



# DBRR 指标
LOAN = ['FED', 'PP']
GROUP = ['UG', 'UGCOMP', 'UGNOCOMP', 'UGUNK', 'GR', 'GRCOMP', 'GRNOCOMP']
YEAR = ['1', '4', '5', '10', '20']
METRICS = ['RT', 'N']
#weight权重可以改
LOAN_weight = {
    "FED": 1,
    "PP": 0
}
GROUP_weight = {
    "UG": 1,
    "GR": 1,
    "UGCOMP": 1,
    "UGNOCOMP": 0,
    "UGUNK": 0,
    "GRCOMP": 1,
    "GRNOCOMP": 0
}
DBRR_index_by_type = np.zeros(n)
DBRR_index = np.zeros(n)
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


#这个就是最终指标
Final_index = np.zeros(n)
for i in range(n):
    Final_index[i] = (DBRR_index[i]+10)/(DBRR_index[i]+20)/(1-CDR_index[i])

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
_HIGHDEG=wash_data(HIGHDEG,HIGHDEG.mean())
X_data.append(_HIGHDEG)

HCM2 = df['HCM2'].values
_HCM2=wash_data(HCM2,HCM2.mean())
X_data.append(_HCM2)

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
        _column1=wash_str_data(column1.values)
        column2=df[name2]
        _column2=wash_str_data(column2.values)
        column3=df[name3]
        _column3=wash_str_data(column3.values)
        column4=df[name4]
        _column4=wash_str_data(column4.values)
        WDRAW_index =_column1+_column2+WDRAW_index
        COMP_index =_column3+_column4+COMP_index
X_data.append(WDRAW_index)
X_data.append(COMP_index)


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
y=Final_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=999)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("line regression")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()

from sklearn.svm import SVR
model=SVR(kernel='poly')
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("svm with kernel poly")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()

model=SVR(kernel='linear')
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("svm with kernel linear")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("knn with k=5")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()


model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("knn with k=10")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()



from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
X=X_data_final.T
y=Final_index
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=999)
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("decision tree")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(random_state=0, n_estimators=100,loss='square')
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("adaboost")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_depth=15, random_state=0)
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predict=model.predict(X_test)
y_model_pred=model.predict(X_train)
plt.scatter(y_test,y_predict,color="y",s=10,  label="train data:R^2={:.4}".format(train_score))
plt.scatter(y_train,y_model_pred,color="r",s=10,  label="test data:R^2={:.4}".format(test_score))
plt.plot([0,1],[0,1],color="b",linestyle="--", linewidth=1)
plt.title("randomforest max_depth=15")
plt.xlabel("true")
plt.ylabel("predict")
plt.legend(loc='lower right')
plt.show()

shows=list(CDR_index)
shows.sort()
plt.bar(range(n),shows)
plt.title("CDR")
plt.show()

shows=list(DBRR_index)
shows.sort()
plt.bar(range(n),shows)
plt.title("DBRR")
plt.show()