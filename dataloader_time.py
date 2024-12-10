import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset,TensorDataset
from sklearn.model_selection import train_test_split
def load_dataset_df():
    labeldata = pd.read_csv(r'.\\bgp_event.csv')

    labeldata['Time'] = pd.to_datetime(labeldata['Time'], format="%Y/%m/%d %H:%M")-pd.Timedelta(hours=0.5)
    labeldata['endTime'] = labeldata['Time'] + pd.Timedelta(hours=2)
    type = labeldata['Type']
    dataspath = r'.\secfivefea\data\abnormal'
    evaidatas = pd.DataFrame()
    fi_list =  []
    for datas in os.listdir(dataspath)[1:]:
        filepath = os.path.join(dataspath, datas)
        fcatch = []



        tendsdata = pd.read_csv(filepath)
        # extra_cols = [col for col in ['ED_9', 'ED_10'] if col in tendsdata.columns]

        # 如果存在，删除它们
        if len(tendsdata.columns) > 56:
            tendsdata = tendsdata.drop(columns=['ED_9', 'ED_10'],axis=1)

        #     print(tendsdata)
        data_error_index = []
        #     tendsdata = pd.DataFrame(tendsdata)
        for ii in range(tendsdata.shape[0]):

            data_col = tendsdata.loc[ii, 'vol_total_num']
            # 1.我们利用float()函数检测是否是float类型，如果非floatl类型，则打印：数据有误，
            # 反之，返回float类型数值。
            try:
                value = int(data_col)
            except:
                value = 'error'
            # 2.利用find()检测是否存在：数据有误。如果有，则将index写入data_error_index中；
            if str(value).find('error') > -1:
                data_error_index.append(ii)

        # 3.drop()函数中利用labels参数，将其删除。
        tendsdata.drop(labels=data_error_index, inplace=True)

        #     data_error_index
        #     datetime = pd.to_datetime((tendsdata['timestamp']),format="%Y-%m-%d %H:%M")

        datetime = pd.to_datetime((tendsdata['timestamp']))
        #     try:

        #     except:
        #         print("errotime:",tendsdata['timestamp'])
        tendsdata['timestamp'] = datetime
        tendsdata.loc[:, 'class'] = 1
        tendsdata.loc[:, 'type'] = 0
        tendsdata.drop_duplicates(subset=['timestamp'], keep='first')
        evaidatas = evaidatas._append(tendsdata, ignore_index=True)
        fi_list.append(tendsdata)
        print("eva", datas, data_error_index, evaidatas.shape)
        # evaidatas.drop_duplicates(subset=['timestamp'], keep='first')
        # print("evafine", datas, data_error_index, evaidatas.shape)

    for index, row in labeldata.iterrows():
        try:
            condition = (evaidatas['timestamp'] >= row['Time']) & (evaidatas['timestamp'] <= row['endTime'])
            evaidatas.loc[condition, 'class'] = 0
            if (row['Type'] == "leak"):
                evaidatas.loc[condition, 'type'] = 1
            elif row['Type'] == "Hijack":
                evaidatas.loc[condition, 'type'] = 2
            elif row['Type'] == "Misconfiguration":
                evaidatas.loc[condition, 'type'] = 3
        except:
            print ("nosuch_time:",evaidatas['timestamp'][0])


    for i,f_df in enumerate(fi_list):
        for index, row in labeldata.iterrows():
            try:
                condition = (f_df["timestamp"]>=row["Time"]) & (f_df["timestamp"]<=row["endTime"])
                if (row["Type"]) == "leak":
                    fi_list[i].loc[condition, "type"] = 1
                elif row["Type"] == "Hijack":
                    fi_list[i].loc[condition, "type"] = 2
                elif row["Type"] == "Misconfiguration":
                    fi_list[i].loc[condition, 'type'] = 3
            except:
                print("next!")
        fi_list[i]["timestamp"] = pd.to_datetime(f_df["timestamp"]).astype('int64')
        fi_list[i] = fi_list[i].fillna(value=0)
        fi_list[i] = fi_list[i].astype(float)
        condition = fi_list[i]['type'] == 0
        subtend = fi_list[i][condition]
        abmsubtend = fi_list[i][condition == False]
        print("normal_tate:",abmsubtend.shape)



    condition = evaidatas['type']==0
    subtend = evaidatas[condition].sample(frac = 0.5)
    abmsubtend = evaidatas[condition==False]
    print(abmsubtend.shape,subtend.shape)
    # tendsdata = pd.concat([subtend,abmsubtend])
    # aidatas = tendsdata.reset_index()
    dataset = evaidatas.fillna(value=0)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp']).astype('int64')
    dataset = dataset.astype(float)
    return dataset,fi_list




class DataFrameToTensor:
    def __init__(self, dataframe,data_list,label_column,window_size):
        self.datasize = len(dataframe)
        self.list_data,self.list_time,self.list_label = self.data_list(data_list,label_column)
        # self.list_label = self.label_list(data_list,label_column)
        self.clabels = label_column


        data = dataframe.drop( label_column,axis=1)  # 删除标签列
        self.data = data.drop(['time_bin','timestamp'],axis=1)
        # self.data = data.drop(['time_bin'],axis=1)
        self.datatime = dataframe['timestamp'].values

        self.labels = dataframe[label_column[1]].values #label_column[1]作为label
        self.data_tensor = None
        self.label_tensor = None
        self.window_size = window_size

    def data_list(self,datalist,label_column):
        data_list = []
        time_list = []
        label_list = []
        for i,f in enumerate(datalist):
            # data = f.drop(label_column,axis = 1)
            data = f.drop(['time_bin'],axis=1)
            time = data['timestamp'].values
            label = data[label_column[1]].values
            data_list.append(data)
            time_list.append(time)
            label_list.append(label)
        return data_list,time_list,label_list

    def label_list(self,flist,label_column):
        list_label = []

        for i,f in enumerate(flist):
            label = f[label_column[1]].values
            list_label.append(label)
        return list_label

    # slice计算函数，可以计算lists 中长度为m的第step个slice
    def splitlstm(self,lists, m, step):

        if step>len(lists)-m+1 :
            raise Exception("TOO far from start" )
        target = step+m-2
        behind = step-1
        if target>len(lists)-1  :
            raise Exception("slice too larg beyond!",target)
        if behind<0:
            raise  Exception("slice too low!",behind)
        resslice = lists.iloc[behind:target+1,:]
        # time = resslice.loc[-1,"timestamp"]
        # resslice = resslice.drop(['timestamp'],axis =1)

        return resslice
        # for i in range(len(lists) - m):
        #     a = lists[i:i + m]
        #
        # return np.array( )
    def loadslice(self,dataset):
        # 为每一个需要分类或训练的数据构造它的slice，使这个slice可以输入到LSTM中，进行训练或预测
        slice_list = pd.DataFrame(columns=['dataframe'])

        for i in  range(1,dataset.shape[0]-self.window_size+2):


             slice = self.splitlstm(dataset ,m=self.window_size,step=i)

             temp_df = pd.DataFrame({'dataframe': [slice]})
             # 将这个包含单个元素的DataFrame连接到result_df中
             slice_list = pd.concat([slice_list, temp_df], ignore_index=True)
             # time_list.append(time)

        return slice_list

    def split_time(self,slice_list,test_size ):
        slice_list['time'] = slice_list['dataframe'].apply(self.select_time)
        # 按时间排序
        slice_list = slice_list.sort_values(by='time')
        # 计算测试集的大小（最新的10%）
        test_size = int(test_size* len(slice_list))
        # 划分训练集和测试集
        data_train = slice_list.iloc[:-test_size]
        data_test = slice_list.iloc[-test_size:]

        return data_train, data_test

    def select_time(self,df):
        # 返回最后一行的列
        return df.iloc[-1]['timestamp']

    def select_type(self,df):
        return df.iloc[-1][self.clabels[1] ]

    def isabnomal(self,df):
        return df.iloc[-1]["type"]

    def drop_columns(self,df):
        # 使用 drop 方法删除指定的列
        df = df.drop(columns=self.clabels)

        return df.drop(columns = ["timestamp"] )
    def splittest(self, test_size,random_state,sample_frac,by_time = False):
        # data_slice_list = self.loadslice(self.data)
        # time_slice_list = self.loadslice(self.datatime)
        # label_slice_list = self.loadslice(self.labels)
        flist = self.list_data
        # llist = self.list_label
        slice_list = pd.DataFrame(columns=['dataframe'])

        for i, f in enumerate(flist):
            data_slice = self.loadslice(f)

            slice_list = pd.concat([slice_list,data_slice],ignore_index=True)
        # slice_list = pd.DataFrame(slice_list)
        # condition = slice_list['dataframe'].iloc[-1]['type']==0
        # subtend = slice_list[condition]
        # abmsubtend = slice_list[condition == False]


        # 根据 'class' 值为 0 和不为 0 将数据分成两个子集
        slice_list['time'] = slice_list['dataframe'].apply(self.select_time)
        if by_time:
            data_train, data_test = self.split_time(slice_list=slice_list,test_size=test_size)
        else:
            data_train,data_test = train_test_split(slice_list,test_size=test_size,random_state=random_state)
        class_groups = data_train['dataframe'].apply(self.isabnomal)
        n_subset = data_train[class_groups == 0]
        ab_subset = data_train[class_groups != 0]
        if n_subset.shape[0] >= ab_subset.shape[0] * sample_frac:
            frac = (ab_subset.shape[0] * sample_frac) / n_subset.shape[0]
            n_subset = n_subset.sample(frac=frac, random_state=42)
            data_train = pd.concat([n_subset, ab_subset])
        train_timestamp = data_train['dataframe'].apply(self.select_time)
        # train_timestamp = pd.DataFrame(train_timestamp, columns=['time'])
        train_y = data_train['dataframe'].apply(self.select_type)

        data_train['dataframe'] = data_train["dataframe"].apply(self.drop_columns)

        test_timestamp = data_test['dataframe'].apply(self.select_time)
        # test_timestamp = pd.DataFrame(test_timestamp, columns=['time'])
        test_y = data_test['dataframe'].apply(self.select_type)

        data_test['dataframe'] = data_test["dataframe"].apply(self.drop_columns)

        # 将结果转换为DataFrame
        result_df = pd.DataFrame(train_timestamp, columns=['last_value'])

        # train_y = data_train["dataframe"].loc[-1,self.clabels[1]]
        # train_timestamp = data_train.iloc[:].loc[-1,"timestamp"]
        # train_x = data_train.iloc[:].drop(columns = ["timestamp",self.clabels])
        # test_y = data_test.iloc[:].loc[-1,self.clabels[1]]
        # test_timestamp = data_test.iloc[:].loc[-1,"timestamp"]
        # test_x = data_test.iloc[:].drop(columns  = ["timestamp",self.clabels])

        # train_x,test_x,train_y,test_y = train_test_split(self.data,self.labels,test_size=test_size,random_state=random_state,shuffle=False)
        # train_x_slice,train_time = self.loadslice(train_x)
        # train_y = train_y[self.window_size-1:]
        # test_x_slice,test_time = self.loadslice(test_x)
        # test_y = test_y[self.window_size-1:]
        print("shapes%%%%%%%%:",data_test.shape,data_train.shape,train_y.shape,test_y.shape)
        return data_train,data_test,train_y,test_y,train_timestamp,test_timestamp
    def convert_to_tensors(self,data,label):
        # 将DataFrame数据转换为PyTorch张量
        # t = type(data)
        # a = data.shape
        data_tensor = [torch.tensor(df.values ) for df in data['dataframe']]
        data_tensor = torch.stack(data_tensor)
        label_tensor = torch.tensor( label.values , dtype=torch.int64)
        return data_tensor,label_tensor
    def create_data_loader(self,data_tensor,label_tensor, batch_size):
        # 使用TensorDataset和DataLoader创建数据加载器
        dataset = TensorDataset( data_tensor,  label_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader
    # def split_to_loader(self,train_x,test_x,train_y,test_y):
    #     train_x_t,train_y_t = self.convert_to_tensors(train_x,train_y)
    #     test_x_t,test_y_t = self.convert_to_tensors(test_x,test_y)
    #     train_dataloader = self.create_data_loader(train_x_t,train_y_t,batch_size=32)
    #     test_dataloader = self.create_data_loader(test_x_t,test_y,batch_size=32)
    #     return train_dataloader,test_dataloader

    def split_to_loader(self, train_x, test_x, train_y, test_y,batch_size):
        train_x_t, train_y_t = self.convert_to_tensors(train_x, train_y)
        test_x_t, test_y_t = self.convert_to_tensors(test_x, test_y)  # 修正此处的test_y
        train_dataloader = self.create_data_loader(train_x_t, train_y_t, batch_size=batch_size)
        test_dataloader = self.create_data_loader(test_x_t, test_y_t, batch_size=batch_size)  # 修正此处的test_y_t
        return train_dataloader, test_dataloader







