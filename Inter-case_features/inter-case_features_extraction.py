#!/usr/bin/env python
# coding: utf-8

"""

# # Inter-Case features:
                        This script is meant to extract different inter-case features
#  
1. Work load features:
                        1.1. **nr_cases:** Number of cases that have been started and did not finish.
                        2.1. **nr_past_events:** Number of events that have occurred in the past X miniuites/hours.
                        3.1. **nr_ongoing_cases:**
                            * let $C$ be an ongoing case,
                            * let $e$ be the last event observed in case $C$.
                            * `feature:` number of other ongoing cases where the last event is E.
                         
2. Demand intensity features:
        2.1. **Case creation intensity:** How many new cases were created since the current case started (divided by the number of seconds since current case was created).
        2.2. **Case completion intensity:** How many cases have completed since this case was created (divided by the number of seconds since current case was created).

3. Temporal contextual features:
                3.1. **tmp_contextual:** Current time's hour of the day (0-23), and day of the week (1-7), and possibly month of the year (1-12). This is meant to capture circadian cycles.
                
                
USAGA: 
        python   inter-case_features_extraction.py

Author:
        Mahmoud Kamel Shoush
        mahmoud.shoush@ut.ee

"""
import os
os.system('pwd')

# ### 1. nr_cases

def add(group):
    df = pd.DataFrame(group)    
    df['en_event'] = df['event_nr'].count()
    return df

def read_add_nr_cases(df, activity_col, time_stamp_col):
    df = df    
    # add en_event for each case
    ddf = df.groupby('Case ID').apply(add)    
    df['en_event'] = ddf['en_event']
    
    # add empty column (nr_cases) 
    df['nr_cases'] = ""
    
    
    previous = [] # track previous cases
    counter = 0 # number of cases    
           
    # get indicies for caseid, event_nr, and en_event
    event_nr = df.columns.get_loc("event_nr")
    en_event = df.columns.get_loc("en_event")
    caseid = df.columns.get_loc("Case ID")
    
    # iterating over rows using iterrows() function 
    for i, j in df.iterrows():      
        if j[caseid] not in previous: # first time of case 
            previous.append(j[caseid])       
            counter +=1
            df.loc[i, 'nr_cases'] =  counter        

        elif j[caseid] in previous and j[event_nr]<j[en_event]: # not first time and not last time for cases
            df.loc[i, 'nr_cases'] = counter

        elif j[caseid] in previous and j[event_nr]==j[en_event]: # last time for cases 
            df.loc[i, 'nr_cases'] = counter -1
            counter = counter -1
    return df




import datetime
import pandas as pd

def get_diff(date1, date2):
    
    #.%f
    diff = datetime.datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')\
    - datetime.datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')
    #diff = datetime.datetime.strptime(date1, '%Y-%m-%d %H:%M:%S.%f')    - datetime.datetime.strptime(date2, '%Y-%m-%d %H:%M:%S.%f')

    return diff.seconds/(60*60)

    
def apply_gr(group, time_stamp_col):
    df = pd.DataFrame(group)
    idx = list(df.index)
    df.index = pd.RangeIndex(len(df.index))
  
    

    max_index = df.shape[0]  
    #print(max_index)
    for index, row in df.iterrows(): 
        
        if index == 0:
            #print(index)
            df.loc[index, 'diff_time'] = 0
            prev =  df[time_stamp_col].iloc[index]
            #print(prev, "\n")


        elif index >0 and index<=max_index-1:
            new_date = df[time_stamp_col].iloc[index]
            tmp = prev
            diff = get_diff(new_date, tmp)
            df.loc[index, 'diff_time'] = diff
            prev = df[time_stamp_col].iloc[index]
    df.index = idx
    return df



## nr_past events
def nr_past_events(group):
    df = pd.DataFrame(group)
    idx = list(df.index)
    df.index = pd.RangeIndex(len(df.index))
 
    max_index = df.shape[0]  
    for index, row in df.iterrows(): 
       
        if index == 0:
            df.loc[index, 'nr_past_events'] = int(0)
            prev_nr =  df['event_nr'].iloc[index]
            prev_time = df['diff_time'].iloc[index]


        elif index >0 and index<=max_index-1:
            current_event_nr = df['event_nr'].iloc[index]
            current_time = df['diff_time'].iloc[index]

            if abs(current_time-prev_time) > 6:
                df.loc[index, 'nr_past_events'] = int(df['event_nr'].iloc[index] -1)
                prev_time = df['diff_time'].iloc[index]
                prev_nr = df['event_nr'].iloc[index]
                
            elif abs(current_time-prev_time) < 6:
                prev_time = df['diff_time'].iloc[index]
                df.loc[index, 'nr_past_events'] = int(prev_nr -1)
      
    df.index = idx
    return df
    

def get_nr_past_events(df,time_stamp_col):
    df = df
    df['diff_time']=""
    
    df = df.groupby("Case ID", as_index=False).apply(apply_gr, time_stamp_col=time_stamp_col).reset_index(drop=True)
    df = df.groupby("Case ID", as_index=False).apply(nr_past_events).reset_index(drop=True)
    # Delete the "Area" column from the dataframe
    df = df.drop("diff_time", axis=1)
    
    return df 
    
    

# 3. **nr_ongoing_cases:** 
#     * let $C$ be an ongoing case,
#     * let $e$ be the last event observed in case $C$.
#     * `feature:` number of other ongoing cases where the last event is E.
#     


import datetime
import pandas as pd


def add_dict(df, activity_col):
    def get_dict(group, activity_col):
        df = pd.DataFrame(group)
        event_nr = df['event_nr'].count()
        caseid = list(df['Case ID'])
        last_event = list(df.loc[df['event_nr'] == event_nr, activity_col])

        return caseid, last_event


    mydict={} # dict {"caseid": "last_event"}
    for i in list(df.groupby("Case ID", as_index=False).apply(get_dict, activity_col)):
        #print(i)
        caseid = i[0][0]
        #print(caseid)
        if not i[1]:
            #continue
            last_event = 'other'
            #print(last_event)
        else:
            last_event = i[1][0]
        #print(last_event)
        mydict[caseid]=last_event
        #break
    return mydict

def nr_ongoing_cases(group, activity_col, mydict):
    df = pd.DataFrame(group)
    event_nr = df['event_nr'].count()
    caseid = list(df['Case ID'])
    last_event = list(df.loc[df['event_nr'] == event_nr, activity_col])
    if not last_event:
        last_event='other'
  
    counter =0
    ll = []
    for key, value in mydict.items():
       
        if key == caseid[0]:
            continue
        else:
            if value == last_event[0]:
                counter +=1
                df['nr_ongoing_cases']=counter
            else:
                counter = counter
                df['nr_ongoing_cases']=counter

    
    return df
    

def get_nr_ongoing_cases(df, activity_col):
    #df = pd.read_csv(df, sep=';')
    df = df
    df['nr_ongoing_cases']=""
    mydict = add_dict(df, activity_col)
      
    df = df.groupby("Case ID", as_index=False).apply(nr_ongoing_cases, activity_col, mydict).reset_index(drop=True)
    
    return df 
    
    




# ## Demand intensity features:
# 
# 
# 4. **Case creation intensity:** How many new cases were created since the current case started (divided by the number of seconds since current case was created).
# 
# 



import pandas as pd

def case_creation_intensity(df):
    #df= pd.read_csv(df, sep=';')
    df = df
    df['cr_cases_intensity']=""
    
    
    def get_dict2(group):
        df = pd.DataFrame(group)
        event_nr = df['event_nr'].count()
        caseid = list(df['Case ID'])

        return caseid, event_nr , 1
    

    mydict2={} # dict {"caseid": ["last_event". first_event]}
    for i in list(df.groupby("Case ID", as_index=False).apply(get_dict2)):
        caseid = i[0][0]
        last_event = i[1]
        first_event = i[2]
        mydict2[caseid]=[last_event, first_event]
        
    for case_dict in mydict2.keys(): 
        start_index = df.loc[(df["Case ID"]==case_dict) & (df["event_nr"]<= mydict2[case_dict][0] ), :].index[0]
        nn = df.loc[(df["Case ID"]==case_dict) & (df["event_nr"]<= mydict2[case_dict][0]), :].shape[0]
        end_index = df.loc[(df["Case ID"]==case_dict) & (df["event_nr"]<= mydict2[case_dict][0]), :].index[nn-3]


        newdf = df.iloc[start_index:end_index+1]

        counter =0
        prev= []
        for index, row in newdf.iterrows():
            case = df['Case ID'].iloc[index]
            event_nr= df['event_nr'].iloc[index]
            if case == case_dict and event_nr==1:
                df.loc[index,'cr_cases_intensity'] = 0
            elif case != case_dict and event_nr!=mydict2[case][0] and case not in prev:
                prev.append(case)
                counter+=1
                #print(f"case: {case},\ncounter: {counter}")
            else:
                if case == case_dict :
                    if counter==0 or df.loc[index,'timesincelastevent']==0:
                        df.loc[index,'cr_cases_intensity'] = counter
                    else:
                        df.loc[index,'cr_cases_intensity'] = counter/df.loc[index,'timesincelastevent']
                else:
                    pass
                
    return df 

    


 # 5. **Case completion intensity:** How many cases have completed since this case was created (divided by the number of seconds since current case was created).
#     


import pandas as pd

def case_completion_intensity(df):
    #df = pd.read_csv(df,sep=';')
    df = df
    df['complete_cases_intensity']=""
    
    
    def get_dict2(group):
        df = pd.DataFrame(group)
        event_nr = df['event_nr'].count()
        caseid = list(df['Case ID'])

        return caseid, event_nr, 1
    

    mydict2={} # dict {"caseid": ["last_event". first_event]}
    for i in list(df.groupby("Case ID", as_index=False).apply(get_dict2)):
        caseid = i[0][0]
        last_event = i[1]
        first_event = i[2]
        mydict2[caseid]=[last_event, first_event]
        
    for case_dict in mydict2.keys():

        start_index = df.loc[(df["Case ID"]==case_dict) & (df["event_nr"]<= mydict2[case_dict][0] ), :].index[0]
        nn = df.loc[(df["Case ID"]==case_dict) & (df["event_nr"]<= mydict2[case_dict][0]), :].shape[0]

        end_index = df.loc[(df["Case ID"]==case_dict) & (df["event_nr"]<= mydict2[case_dict][0]), :].index[nn-3]

        newdf = df.iloc[start_index:end_index+1]

        counter =0
        prev = []
        for index, row in newdf.iterrows():
            case = df['Case ID'].iloc[index]
            event_nr= df['event_nr'].iloc[index]
            
            if case == case_dict and event_nr==1: # case started
                df.loc[index,'complete_cases_intensity'] = 0
                
            elif case !=case_dict and event_nr==mydict2[case][0]: # between
                counter+=1
                pass
                
            elif case == case_dict and event_nr!=1: # case started
                if counter ==0 or df.loc[index,'timesincecasestart']==0:
                    df.loc[index,'complete_cases_intensity'] = counter
                else:
                    df.loc[index,'complete_cases_intensity'] = counter/df.loc[index,'timesincecasestart']
                    
                
                
    return df 

    


# ## Temporal contextual features:
# 
# **6. tmp_contextual:** Current time's hour of the day (0-23), and day of the week (1-7), and possibly month of the year (1-12). This is meant to capture circadian cycles.


import pandas as pd
import datetime

def add_tmp_features(df, time_stamp_col):
    df = df
    
    def get_dict2(group):
        df = pd.DataFrame(group)
        event_nr = df['event_nr'].count()
        caseid = list(df['Case ID'])

        return caseid, event_nr , 1
    

    mydict2={} # dict {"caseid": ["last_event". first_event]}
    for i in list(df.groupby("Case ID", as_index=False).apply(get_dict2)):
        caseid = i[0][0]
        last_event = i[1]
        first_event = i[2]
        mydict2[caseid]=[last_event, first_event]
        
    for case_dict in mydict2.keys():
        date = list(df[df["event_nr"]==mydict2[case_dict][0]][time_stamp_col])[0]
        #print(date)
        ff ='%Y-%m-%d %H:%M:%S'
        #f = '%Y-%m-%d %H:%M:%S.%f'
        dt = datetime.datetime.strptime(date, ff)

        hour_day = dt.hour # (0:23)
        day_week = dt.weekday()+1
        day_month = dt.day
        month_year = dt.month

        df.loc[df["Case ID"]==case_dict, "hour_day"]=hour_day
        df.loc[df["Case ID"]==case_dict,'day_week']=day_week
        df.loc[df["Case ID"]==case_dict,'day_month']=day_month
        df.loc[df["Case ID"]==case_dict,'month_year']=month_year
        
    return df





import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#Let's make this notebook reproducible, you can use any number ex = 42
np.random.seed(42)

import datetime
import re


def run_experiments(data):
    print(f"\n================================={data}=========================\n")
    df = pd.read_csv(data, sep=';')


    #for data in file:
    if re.search('BPIC11_f*?', data) or re.search('BPIC15_*?', data)            or re.search('BPIC17*?',data) or re.search('sepsis_cases*?', data):
        time_stamp_col = 'time:timestamp'
    else:
        time_stamp_col= 'Complete Timestamp'

    if re.search('BPIC11_f*?', data):
        activity_col = 'Activity code'
    else:
        activity_col = 'Activity'

    # 1st feature add nr_cases
    print("\nFirst Feature: nr_cases")
    df = read_add_nr_cases(df, activity_col, time_stamp_col)
    #print(df.info())
    # 2nd nr_past_events
    print("\nSecond Feature: nr_past_events")
    df = get_nr_past_events(df, time_stamp_col)
    #print(df.info())

    #3rd: nr_ongoing_case
    print("\nThird Feature: nr_ongoing_cases")
    df = get_nr_ongoing_cases(df, activity_col)
    #print(df.info())
    
    # 4th: nr_ongoing_case
    print("\nFourth Feature: case_creation_intensity")
    df = case_creation_intensity(df)
    #print(df.info())
    
    # 5th: nr_ongoing_case
    print("\nFifth Feature: case_completeion_intensity")
    df = case_completion_intensity(df)
    #print(df.info())
    
    # 6th: tmp_features
    print("\nSixth Feature: Tmp_features")
    df = add_tmp_features(df, time_stamp_col)
    #print(df.info()) 
    return df 
    
    
    
 
import os
import glob

path = './labeled_logs_csv_processed/'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)

#result = ['traffic_fines_1.csv',]

    
for data in result:

    
     df = run_experiments(data)
     df.to_csv("./inter_"+data, sep=";",    index=False)


