# import libraties
import pandas as pd
import numpy as np
import sys
import pickle
# https://pm4py.fit.fraunhofer.de/documentation
import pm4py
from pm4py.objects.log.util.log import project_traces
from pm4py.objects.log.util import interval_lifecycle
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py.objects.log.obj import EventLog, Trace
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


# function to project the trace
def project_nth(log, index):
    print(str(project_traces(log)[index]))


# function for data preprocessing
def data_preprocessing(trace):  
    # remove columns where missing > 30%
    # get persentage of missing values in a column
    percent_missing = trace.isnull().sum() * 100 / len(trace)

    # extract list of coumns which are more then 30%
    miss_ls = list(percent_missing[percent_missing>30].keys())

    # keep the remaining columns where missing data is less then 30%
    trace = trace.loc[:, ~trace.columns.isin(miss_ls)]

    # adding time features month and year
    trace['startTime'] = pd.to_datetime(trace['startTime'])
    trace['completeTime'] = pd.to_datetime(trace['completeTime'])
    trace['year'] = trace['startTime'].dt.year
    trace['month'] = trace['startTime'].dt.month

    # converting dataframe to event log
    trace_log = pm4py.format_dataframe(trace, case_id='case', activity_key='event', timestamp_key='completeTime', start_timestamp_key='startTime')
    trace_log = pm4py.convert_to_event_log(trace_log)

    # add other time features
    trace_log = interval_lifecycle.assign_lead_cycle_time(trace_log)
 
    # extract traces only till the decleration is rejected(included), otherwise include it payment handled 
    prefix_traces = []
    for trace in trace_log:
        trace_end_flag = False
        for i,event in enumerate(trace):
            if "Declaration REJECTED" in event['event']:
                trace_end_flag = True
                i+=1
                break
            if "Payment Handled" in event['event']:
                trace_end_flag = True
        if trace_end_flag:
            prefix_traces.append(Trace(trace[:i], attributes = trace.attributes))
    prefix_traces = EventLog(prefix_traces)

    # convert extracted traces to dataframe
    trace = pm4py.convert_to_dataframe(prefix_traces)
    trace.head()
    return trace

# function for train test validation split
def train_test_split(trace):
    # get completion time and sort
    completion_time_ls = list(trace.groupby(['case'])['completeTime'].max())
    completion_time_ls = sorted(completion_time_ls)


    # split on 70%  and 85% of max time 
    train_split_portion = 0.70
    val_split_seperation = 0.85
    total_data = len(completion_time_ls)
    train_len = int(train_split_portion*total_data)
    val_len = int(val_split_seperation*total_data)
    last_train_completion_time = completion_time_ls[train_len]
    last_val_completion_time = completion_time_ls[val_len]
    val_start_time = last_train_completion_time
    print(f"train test and validation split times are - {last_train_completion_time} , {last_val_completion_time} ")

    # take all traces where start dates are after the last_train_completion_time
    dtype_list = list(trace.dtypes) # get original types of the columns
    train_df = pd.DataFrame(columns = trace.columns)
    test_df = pd.DataFrame(columns = trace.columns)
    val_df = pd.DataFrame(columns = trace.columns)
    train_count,test_count,val_count = 0,0,0
    intersecting_traces = []
    for name, group in trace.groupby(['case'],as_index=False):
        if group['completeTime'].iloc[-1] <= last_train_completion_time:
            train_df = train_df.append(group)
            train_count+=1
        elif (group['startTime'].iloc[0] >= last_train_completion_time) and (group['completeTime'].iloc[-1] <= last_val_completion_time):
            val_df = val_df.append(group)
            val_count+=1        
        elif group['startTime'].iloc[0] >= last_val_completion_time:
            test_df = test_df.append(group)
            test_count+=1
        else:
            intersecting_traces.append(group)


    # converting train and test to their original data types
    for i,col in enumerate(train_df.columns):
        train_df[col] = train_df[col].astype(dtype_list[i])

    for i,col in enumerate(test_df.columns):
        test_df[col] = test_df[col].astype(dtype_list[i])

    for i,col in enumerate(val_df.columns):
        val_df[col] = val_df[col].astype(dtype_list[i])


    print("train, val and test count")
    print(train_count,val_count,test_count)

    # loss of traces due to temporal intersection
    # these are the traces which started and intersecting with split time - Timestamp('2018-10-15 17:31:12+0000', tz='UTC')
    print(f"loss of traces due to temporal intersection - {len(intersecting_traces)}")
    return train_df, test_df, val_df



# function for extracting prefix traces and our target variable decleration
def extract_prefixtraces_and_decleration(permits,t_length):
    # converting dataframe to event log
    trace_log = pm4py.format_dataframe(permits, case_id='case', activity_key='event', timestamp_key='completeTime', start_timestamp_key='startTime')
    trace_log = pm4py.convert_to_event_log(trace_log)

    # to extract target varaible, 
    # if event starts with the name decleration rejected it is considered as rejected
    declerations = []
    for trace in trace_log:
        flag = False
        for i,event in enumerate(trace):
            if "Declaration REJECTED" in event['event']:
                flag = True
                break
        
        if flag:
            declerations.append(1)
        else:
            declerations.append(0)

    # extract traces only till the decleration is rejected (excluded), otherwise complete trace 
    prefix_traces = []
    for trace in trace_log:
        for i,event in enumerate(trace):
            if "Declaration REJECTED" in event['event']:
                break
        prefix_traces.append(Trace(trace[:i], attributes = trace.attributes))
    prefix_traces = EventLog(prefix_traces)

    # generate prefixes, note that we need to add the casts to EventLog and Trace to make sure that the result is a PM4Py EventLog object
    trace_prefixes = EventLog([Trace(trace[0:t_length], attributes = trace.attributes) for trace in prefix_traces])

    # convert logs to dataframe
    # final base dataframe
    df = pm4py.convert_to_dataframe(trace_prefixes)
    return df, declerations


## Common functions for Encodings 
# function to save the data
def save_data(X,y,feature_names, save_path):
    data_dict = {}
    data_dict['X'] = X
    data_dict['y'] = y
    data_dict['feature_names'] = feature_names

    # save pickle
    with open(save_path, 'wb') as handle:
        pickle.dump(data_dict, handle)


# load the data from pickle
def load_data(load_path):
    with open(load_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


# function to get the one hot encoded vectors of categorical values
def get_ohe_dict(categorical_vars, df):
    ohe_dict = {}
    for var in categorical_vars:
        var_dict = {}
        var_data = sorted(df[var].unique())
        var_len = len(var_data)
        for i,cat in enumerate(var_data):
            var_dict[cat] = [0]*var_len
            var_dict[cat][i] = 1

        ohe_dict[var] = var_dict

    return ohe_dict


# padding function for ohe encoding
def cat_padding(vec, t_length, attr_length):
    desired_length = t_length*attr_length
    vec_length = len(vec)
    if vec_length != desired_length:
        pad_vec = [0]*(desired_length-vec_length)
        vec.extend(pad_vec)
    return vec


# padding function for non-ohe encoding
def num_padding(vec, t_length):
    vec_length = len(vec)
    if vec_length != t_length:
        pad_vec = [0]*(t_length-vec_length)
        vec.extend(pad_vec)
    return vec


# boolean encoding
def boolean_encoding(df,declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length):
# here for each trace we extract ohe vector for activity and sum them up and if count is greater then 1 we make them 1 
# because this encoding only provides info, if the activity was there or not
    data = []
    for id, group in df.groupby(['case:concept:name']):
        feature_vec = []

        # add categorical and numerical event attributes
        for cat_atr in str_ev_attr[:1]:
            attr_length = len(list(ohe_dict[cat_atr].values())[0])
            str_ev_vec = np.array([0]*attr_length)
            for ca in group[cat_atr]:
                str_ev_vec  = str_ev_vec + np.array(ohe_dict[cat_atr][ca])

            # make it a non frequency vector (if count is greater then 1 make it 1)
            for i,num in enumerate(str_ev_vec):
                if num>1:
                    str_ev_vec[i]=1

            feature_vec.extend(list(str_ev_vec))

        data.append(feature_vec)

    # saving data 
    encode_name = 'boolean_encode_'
    save_path = save_path_base + encode_name + df_type +'_trace_len_'+str(t_length)+ '.pickle'
    save_data(data, declerations, ohe_dict ,save_path)


# frequency encoding
def frequency_encoding(df,declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length):
# here for each trace we extract ohe vector for activity and sum them up 
# because this encoding only provides count of how many times the activity appears

    data = []
    for id, group in df.groupby(['case:concept:name']):
        feature_vec = []

        # add categorical and numerical event attributes
        for cat_atr in str_ev_attr[:1]:
            attr_length = len(list(ohe_dict[cat_atr].values())[0])
            str_ev_vec = np.array([0]*attr_length)
            for ca in group[cat_atr]:
                str_ev_vec  = str_ev_vec + np.array(ohe_dict[cat_atr][ca])

            feature_vec.extend(list(str_ev_vec))

        data.append(feature_vec)

    # save results
    encode_name = 'frequency_encode_'
    save_path = save_path_base + encode_name + df_type +'_trace_len_'+str(t_length)+ '.pickle'
    save_data(data, declerations, ohe_dict ,save_path)


# complex index encoding
def complex_index_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length):
    ## Complex index based encoding - static feature (trace attributes) + n events encoding + event features
    # here for each trace we put events encoded in order and there aatributes along with padding to make vector length same
    # similatly for trace attributes but since it is trace attributes that is only for once

    data = []
    for id, group in df.groupby(['case:concept:name']):
        feature_vec = []
        # add categorical and numerical event attributes along with paddings 
        for cat_atr in str_ev_attr:
            str_ev_vec = []
            attr_length = len(list(ohe_dict[cat_atr].values())[0])
            for ca in group[cat_atr]:
                str_ev_vec.extend(ohe_dict[cat_atr][ca])
            
            # padding
            str_ev_vec = cat_padding(str_ev_vec, t_length, attr_length)
            feature_vec.extend(str_ev_vec)

        for num_atr in num_ev_attr:
            num_ev_vec = []
            num_ev_vec.extend(list(group[num_atr]))

            # padding
            num_ev_vec = num_padding(num_ev_vec, t_length)
            feature_vec.extend(num_ev_vec)

        # add categorical and numerical trace attributes
        for num_t_atr in num_tr_attr:
            feature_vec.extend(group[num_t_atr].iloc[0])
        for cat_t_atr in str_tr_attr:
            feature_vec.extend(ohe_dict[cat_t_atr][group[cat_t_atr].iloc[0]])

        # add vector to data
        data.append(feature_vec)

    # check if all vector lengths are same 
    vec_len = len(data[0])
    for i, d in enumerate(data):
        if len(d)!=vec_len:
            print(i, len(d))

    # save results
    encode_name = 'complex_index_encode_'
    save_path = save_path_base + encode_name + df_type +'_trace_len_'+str(t_length)+ '.pickle'
    save_data(data, declerations, ohe_dict ,save_path)


## LSTM encoding
def LSTM_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length):
    # here we create sequence of each trace
    # so the dimentions will be (number of examples * trace_length * feature_length )
    data = []
    for id, group in df.groupby(['case:concept:name']):
        feature_vec = []
        for index, row in group.iterrows():
            row_vec = []
            for cat_atr in str_ev_attr:
                row_vec.extend(ohe_dict[cat_atr][row[cat_atr]])
            for num_atr in num_ev_attr:
                row_vec.append(row[num_atr])

            # add categorical and numerical trace attributes
            for num_t_atr in num_tr_attr:
                row_vec.append(group[num_t_atr].iloc[0])
            for cat_t_atr in str_tr_attr:
                row_vec.extend(ohe_dict[cat_t_atr][group[cat_t_atr].iloc[0]])

            feature_vec.append(row_vec)
        
        # add vector to data
        data.append(feature_vec)

    # converting to array
    data = np.array([np.array(ls) for ls in data])

    # shape we want for all the traces
    feature_len = len(data[0][0])
    desired_shape = (t_length,feature_len)
    desired_shape

    # padding data to make equal shape of vectors
    padded_data = []
    for case in data:
        pd_case = np.zeros(desired_shape)
        pd_case[:case.shape[0],:case.shape[1]] = case
        padded_data.append(pd_case)
    padded_data = np.array(padded_data)
    padded_data.shape

    # save results
    encode_name = 'lstm_encode_'
    save_path = save_path_base + encode_name + df_type +'_trace_len_'+str(t_length)+ '.pickle'
    save_data(padded_data, declerations, ohe_dict ,save_path)




########==============================================   main ==============================================########
def main(prefix_length=10, dir_load_path = 'data/BPIC2020_CSV/filterd_TravelPermits.csv', dir_save_path = 'data/training_data/'):

    # read data in csv 
    trace_df = pd.read_csv(dir_load_path)

    # basic preprocessing over travel permits dataframe
    trace_processed = data_preprocessing(trace_df)
    trace_base_df = trace_processed.copy()

    # split data in train and test sets
    train_df, test_df, val_df = train_test_split(trace_processed)

    # define comman variables 
    # ==============================================
    # trace length and saving path
    t_length = prefix_length
    save_path_base = dir_save_path
    
    # passed features we want to extract
    # str_ev_attr	String attributes at the event level: these are hot-encoded into features that may assume value 0 or value 1.
    # str_tr_attr	String attributes at the trace level: these are hot-encoded into features that may assume value 0 or value 1.
    # num_ev_attr	Numeric attributes at the event level: these are encoded by including the last value of the attribute among the events of the trace.
    # num_tr_attr	Numeric attributes at trace level: these are encoded by including the numerical value.
    str_ev_attr = ['concept:name']
    str_tr_attr = ['OrganizationalEntity','month']
    num_ev_attr = ['@@approx_bh_partial_lead_time','@@approx_bh_this_wasted_time']
    num_tr_attr = []
    # ================================================


    # train data preparation 
    df, declerations = extract_prefixtraces_and_decleration(train_df,t_length)
    # create one hot encoding dict fot categorical variables, variables which we want to be one hot encoded
    categorical_vars = str_ev_attr + str_tr_attr
    ohe_dict = get_ohe_dict(categorical_vars, trace_base_df)
    # encode training data and save to base dir 
    df_type = 'train'
    print(f"preparing training data for prefix length {t_length}")
    boolean_encoding(df, declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length)
    frequency_encoding(df, declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length)
    complex_index_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length)
    LSTM_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length)


    # test data preparation 
    df, declerations = extract_prefixtraces_and_decleration(test_df,t_length)
    # encode test data and save to base dir 
    df_type = 'test'
    print(f"preparing test data for prefix length {t_length}")
    boolean_encoding(df, declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length)
    frequency_encoding(df, declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length)
    complex_index_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length)
    LSTM_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length)


    # val data preparation 
    df, declerations = extract_prefixtraces_and_decleration(val_df,t_length)
    # encode val data and save to base dir 
    df_type = 'val'
    print(f"preparing val data for prefix length {t_length}")
    boolean_encoding(df, declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length)
    frequency_encoding(df, declerations,ohe_dict,str_ev_attr,save_path_base,df_type,t_length)
    complex_index_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length)
    LSTM_encoding(df,declerations,ohe_dict,str_ev_attr,str_tr_attr,num_ev_attr,num_tr_attr,save_path_base,df_type,t_length)

    # process completed
    print("Done!")


if __name__=='__main__':

    # extract passed paramter along, default trace length is 10
    try:
        prefix_length = int(sys.argv[1])
        print(f"Encoding data for prefix length {prefix_length}")
    except:
        print("Please enter integer value, using default prefix value 10 and encoding")
        prefix_length = 10

    # location of travel permit raw file 
    dir_load_path = 'data/BPIC2020_CSV/filterd_TravelPermits.csv'

    # location of encoding saving path
    dir_save_path = 'data/training_data/'

    # calling main function with passed implementation method
    main(prefix_length, dir_load_path, dir_save_path)


