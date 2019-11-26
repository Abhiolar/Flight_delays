import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import numpy as np
def clean_data(df):
    df.head()
    del df['Unnamed: 0']
    del df['Unnamed: 0.1']
    df.dropna(inplace=True)
    df['DepTime'] = df['DepTime'].astype(int)
    df['SchedDep'] = df['SchedDep'].astype(int)
    df['DepTime'] = df['DepTime'].astype(str)
    df['SchedDep'] = df['SchedDep'].astype(str)

    df['DepTime'] = df['DepTime'].apply(lambda x: '{0:0>4}'.format(x))#adding zeros to the front to convert it to datetime fromat
    df['SchedDep'] = df['SchedDep'].apply(lambda x: '{0:0>4}'.format(x))
    minus = df[df['SchedDep'].str.contains("-")].index
    df.drop(minus, inplace=True)
    df.reset_index(drop=True)
    df.DepTime = df['DepTime'].apply(lambda x: '{0:0<6}'.format(x))#adding zeros to the back of the values
    df.SchedDep = df['SchedDep'].apply(lambda x: '{0:0<6}'.format(x))
    defect =[]
    for j, i in enumerate (df.SchedDep.str[2]):
        if i == '6':
            defect.append(j)
        if i == '7':
            defect.append(j)
        if i == '8':
            defect.append(j)
        if i == '9':
            defect.append(j)
    df_defect = df.index[defect]
    df.drop(index=df_defect,axis =1, inplace=True)
    df['DepTime'] = pd.to_datetime(df['DepTime'],format= '%H%M%S' )
    df['SchedDep'] = pd.to_datetime(df['SchedDep'],format= '%H%M%S')
    df['Delay_min'] = (df['DepTime'] - df['SchedDep']).dt.seconds/60
    df['Delay_min'] = np.where(df['Delay_min'] > 1400,df['DepDelay'], df['Delay_min'])
    return df
                   