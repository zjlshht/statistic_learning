from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
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
    num = 1e-11
    count = 0
    for i in result:
      if i != -1:
        count += i
        num += 1
    mean = count / num
    result[result == -1] = mean
  else:
    result[result == -1] = 0
  return result


def get_data_from_df(name, replace=False):
  column = df[name].values
  res = clear_nan(column, replace)
  return res


def column_exist(name, column_name):
  if name in column_name:
    return True
  else:
    return False


def corr(x, y):
  up = sum(x * y)
  down = np.sqrt(sum(x**2) * sum(y**2))
  return up / down


def find_index(array, member):
  if array[0] > member:
    for i in range(len(array)):
      if array[i] <= member:
        return i
  else:
    for i in range(len(array)):
      if array[i] >= member:
        return i


def plot_curve(y_test, y_pred, name='roc', threshold=0.5, model=''):
  fpr, tpr, _ = roc_curve(y_test, y_pred)
  if name == 'roc':
    plt.plot(fpr, tpr)
    plt.title("{} curve of {}".format(name, model))
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()
  if name == 'pr':
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_index = find_index(thresholds, threshold)
    Precision = precision[pr_index]
    Recall = recall[pr_index]
    plt.plot(recall, precision)
    if len(recall) < 10:
      plt.scatter(recall, precision, color="r", s=10)
    plt.title("{} curve of {}".format(name, model))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.show()
    return [Precision, Recall]


df = pd.read_csv('MERGED2018_19_PP.csv')
year_number = 18
column_name = df.columns


#CDR
CDR2 = get_data_from_df('CDR2')
CDR3 = get_data_from_df('CDR3')
CDR_index = CDR2 / 2 + CDR3 / 2

n = len(CDR2)


# RPY
years = ['1', '3', '5', '7']
student_type = ['', 'COMPL_', 'HI_INC_', 'MD_INC_', 'LO_INC_']
type_weight = {'': 1, 'COMPL_': 1, "HI_INC_": 0.5, "MD_INC_": 1, "LO_INC_": 2}
RPY_ac_type = np.zeros(n)
RPY_index_ac_by_type = np.zeros(n)

for Type in type_weight:
  RPY_index_ac_by_N = np.zeros(n)
  RPY_ac_N = np.zeros(n)
  for year in years:
    numbers = Type + 'RPY_' + year + 'YR_' + 'N'
    rate = Type + 'RPY_' + year + 'YR_' + 'RT'
    RPY_N = get_data_from_df(numbers, True)
    RPY_RT = get_data_from_df(rate, True)
    RPY_index_ac_by_N += RPY_N * RPY_RT / int(year)
    RPY_ac_N += RPY_N / int(year)
  RPY_index_ac_by_type += RPY_index_ac_by_N * type_weight[Type] / (
    RPY_ac_N + 1e-8)
  RPY_ac_type += type_weight[Type]
RPY_index = RPY_index_ac_by_type / RPY_ac_type


# DBRR
GROUP = ['UG', 'UGCOMP', 'GR', 'GRCOMP']
YEAR = ['1', '4', '5', '10', '20']
DBRR_index = np.zeros(n)

for group in GROUP:
  DBRR_index_up = np.zeros(n)
  DBRR_index_down = np.zeros(n)
  for year in YEAR:
    DBRR_rt_name = 'DBRR' + year + '_' + 'FED' + '_' + group + '_RT'
    DBRR_n_name = 'DBRR' + year + '_' + 'FED' + '_' + group + '_N'
    if column_exist(DBRR_rt_name, column_name):
      DBRR_rt = get_data_from_df(DBRR_rt_name, True)
      DBRR_n = get_data_from_df(DBRR_n_name, True)
    else:
      continue
    DBRR_index_up += DBRR_rt * DBRR_n * int(year)
    DBRR_index_down += DBRR_n * int(year)
  DBRR_index_by_group = DBRR_index_up / (DBRR_index_down + 1e-8)
  DBRR_index += DBRR_index_by_group / 4


# BBRR
GROUP = ['UG', 'UGCOMP', 'GR', 'GRCOMP']
YEAR = ['1', '2']
Status = [
  'DFLT', 'DLNQ', 'FBR', 'DFR', 'NOPROG', 'MAKEPROG', 'PAIDINFULL', 'DISCHARGE'
]
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
    BBRR_name = 'BBRR' + year + '_' + 'FED' + '_' + group + '_'
    BBRR_n_name = BBRR_name + 'N'
    if column_exist(BBRR_n_name, column_name):
      BBRR_n = get_data_from_df(BBRR_n_name, True)
    else:
      continue
    for status in Status:
      BBRR_rt_name = BBRR_name + status
      BBRR_rt = get_data_from_df(BBRR_rt_name, True)
      BBRR_index_by_status += BBRR_rt * Status_weight[status] / 3
    BBRR_index_down += BBRR_n * int(year)
    BBRR_index_up += BBRR_index_by_status * BBRR_n * int(year)
  BBRR_index += BBRR_index_up / (BBRR_index_down + 1e-8) / 4


# DEBT
student_type = ['', 'GRAD_', 'HI_INC_', 'MD_INC_', 'LO_INC_']
type_weight = {'': 1, 'GRAD_': 2, "HI_INC_": 1.5, "MD_INC_": 1, "LO_INC_": 0.5}
DEBT_index_down = np.zeros(n)
DEBT_index_up = np.zeros(n)

for type_name in student_type:
  DEBT_n_name = type_name + 'DEBT_N'
  DEBT_mdn_name = type_name + 'DEBT_MDN'
  DEBT_n = get_data_from_df(DEBT_n_name, True)
  DEBT_mdn = get_data_from_df(DEBT_mdn_name, True)
  DEBT_index_up += DEBT_mdn * DEBT_n * type_weight[type_name]
  DEBT_index_down += DEBT_n * type_weight[type_name]
DEBT_index = DEBT_index_up / DEBT_index_down

fig = plt.figure()
ax = plt.subplot()
ax.boxplot(BBRR_index)
plt.title("BBRR_index in year{}-{}".format(year_number, year_number + 1))

fig = plt.figure()
ax = plt.subplot()
ax.boxplot(DBRR_index)
plt.title("DBRR_index in year{}-{}".format(year_number, year_number + 1))

fig = plt.figure()
ax = plt.subplot()
ax.boxplot(DEBT_index)
plt.title("DEBT_index in year{}-{}".format(year_number, year_number + 1))

fig = plt.figure()
ax = plt.subplot()
ax.boxplot(CDR_index)
plt.title("CDR_index in year{}-{}".format(year_number, year_number + 1))

fig = plt.figure()
ax = plt.subplot()
ax.boxplot(RPY_index)
plt.title("RPY_index in year{}-{}".format(year_number, year_number + 1))

#老模型
'''
BBRR_index[BBRR_index>0.14]=1
BBRR_index[BBRR_index<=0.14]=0
DBRR_index[DBRR_index>1]=1
DBRR_index[DBRR_index<=1]=0
DEBT_index[DEBT_index>25000]=1
DEBT_index[DEBT_index<=25000]=0
CDR_index[CDR_index>0.1]=1
CDR_index[CDR_index<=0.1]=0
RPY_index[RPY_index>0.6]=1
RPY_index[RPY_index<=0.6]=0
Final_index = CDR_index+BBRR_index+DEBT_index+DBRR_index+RPY_index
Final_index[Final_index>0]=1
'''

#新模型
BBRR_index[BBRR_index > 0.13] = 1
BBRR_index[BBRR_index <= 0.13] = 0
DBRR_index[DBRR_index > 0.95] = 1
DBRR_index[DBRR_index <= 0.95] = 0
DEBT_index[DEBT_index > 23000] = 1
DEBT_index[DEBT_index <= 23000] = 0
CDR_index[CDR_index > 0.09] = 1
CDR_index[CDR_index <= 0.09] = 0
RPY_index[RPY_index > 0.6] = 1
RPY_index[RPY_index <= 0.6] = 0
Final_index = CDR_index + BBRR_index + DEBT_index + DBRR_index + RPY_index
Final_index[Final_index > 0] = 1


X_data = []
number_type = [
  'INEXPFTE', 'AVGFACSAL', 'C100_4_POOLED', 'C100_L4_POOLED', 'C200_4_POOLED',
  'C200_L4_POOLED', 'C150_4_POOLED', 'C150_L4_POOLED', 'FTFTPCTPELL',
  'FTFTPCTFLOAN', 'COSTT4_P', 'COSTT4_A', 'TUITIONFEE_IN', 'TUITIONFEE_OUT',
  'TUITIONFEE_PROG', 'TUITFTE'
]
for name in number_type:
  datas = get_data_from_df(name, True)
  X_data.append(datas)

ACT_class = ['CM', 'EN', 'MT', 'WR']
ACT_quan = ['25', 'MID', '75']
for classs in ACT_class:
  for quan in ACT_quan:
    name = 'ACT' + classs + quan
    score = get_data_from_df(name, True)
    X_data.append(score)

Student_class = ['', 'DEP_', 'IND_', 'HI_INC_', 'MD_INC_', 'LO_INC_']
YEAR = ['2', '3', '4', '6', '8']
WDRAW_index = np.zeros(n)
COMP_index = np.zeros(n)
for classs in Student_class:
  for year in YEAR:
    name1 = classs + 'WDRAW' + '_2YR_' + 'TRANS_YR' + year + '_RT'
    name2 = classs + 'WDRAW' + '_4YR_' + 'TRANS_YR' + year + '_RT'
    name3 = classs + 'COMP' + '_2YR_' + 'TRANS_YR' + year + '_RT'
    name4 = classs + 'COMP' + '_4YR_' + 'TRANS_YR' + year + '_RT'
    column1 = get_data_from_df(name1, True)
    column2 = get_data_from_df(name2, True)
    column3 = get_data_from_df(name3, True)
    column4 = get_data_from_df(name4, True)
    X_data.append(column1)
    X_data.append(column2)
    X_data.append(column3)
    X_data.append(column4)

MTHCMP_type = ['1', '2', '3', '4', '5', '6']
for types in MTHCMP_type:
  name = 'MTHCMP' + types
  MTHCMP = get_data_from_df(name, True)
  X_data.append(MTHCMP)

Tuition_type = ['41', '42', '43', '44', '45', '4_048', '4_3075', '4_75UP', '4']
School_type = ['OTHER', 'PRIV', 'PROG', 'PUB']
for tuition in Tuition_type:
  for _type in School_type:
    name = 'NPT' + tuition + '_' + _type
    column = get_data_from_df(name, True)
    X_data.append(column)

RET_year = ['4', 'L4']
RET_type = ['FT', 'PT']
for year in RET_year:
  for types in RET_type:
    name = 'RET' + '_' + types + year
    column = get_data_from_df(name, True)
    X_data.append(column)

minority_serving = {'HBCU', 'PBI', 'ANNHI', 'AANAPII', 'HSI', 'NANTI'}
Ms = np.zeros(n)
for name in minority_serving:
  ms_type = get_data_from_df(name)
  Ms += ms_type
Ms[Ms > 0] = 1
X_data.append(Ms)
MAIN = get_data_from_df('MAIN')
X_data.append(MAIN)

LATITUDE = get_data_from_df('LATITUDE')
LATITUDE /= 90
X_data.append(LATITUDE)

LONGITUDE = get_data_from_df('LONGITUDE')
LONGITUDE /= 180
X_data.append(LONGITUDE)

HCM2 = get_data_from_df('HCM2')
X_data.append(HCM2)

HIGHDEG = get_data_from_df('HIGHDEG')
X_data.append(HIGHDEG)

OPEFLAG = get_data_from_df('OPEFLAG')
OPEFLAG[OPEFLAG != 1] = 0
X_data.append(OPEFLAG)

X_data_final = np.array(X_data)

X = X_data_final.T
y = Final_index
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.25, random_state=999)

scaler = StandardScaler()
scaler.fit(X_train[:, :194])
X_train[:, :194] = scaler.transform(X_train[:, :194])
X_test[:, :194] = scaler.transform(X_test[:, :194])

clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(
  y_test, y_pred, name='pr', model='RandomForest with max tree number=50')

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(
  y_test, y_pred, name='pr', model='RandomForest with max tree number=100')

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
plot_curve(y_test, y_pred, name='pr', model='LogisticRegression')

clf = SVR(kernel='poly', degree=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(y_test, y_pred, name='pr', model='SVM with poly in ten degree')

clf = SVR(kernel='poly', degree=6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(y_test, y_pred, name='pr', model='SVM with poly in six degree')

clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(y_test, y_pred, name='pr', model='DecisionTree')

clf = AdaBoostRegressor(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(
  y_test, y_pred, name='pr', model='AdaBoostRegressor with max tree number=100')

clf = AdaBoostRegressor(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(
  y_test, y_pred, name='pr', model='AdaBoostRegressor with max tree number=50')

clf = AdaBoostRegressor(n_estimators=150)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_curve(
  y_test, y_pred, name='pr', model='AdaBoostRegressor with max tree number=150')
