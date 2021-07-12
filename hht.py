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


def clear_nan(column, replace=0):
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


def column_exist(name, column):
    if name in column:
        return True
    else:
        return False
