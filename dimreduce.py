import os
import re
import json
from torch import nn

import pandas as pd
import torch
import random
import numpy as np

from LSTM import SA_LSTM
from bilstm import BISA_LSTM
from comLSTM import comLSTM
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sample_frac = 1.0045536770380648
batch_size = 16
window_size = 49
split_by_time = True

if split_by_time:
    from dataloader_time import DataFrameToTensor,load_dataset_df

    df_data, data_list = load_dataset_df()
    tensor_data = DataFrameToTensor(dataframe=df_data, data_list=data_list, label_column=['class', 'type'],
                                    window_size=window_size)
    train_x, test_x, train_y, test_y, train_time, test_time = tensor_data.splittest(test_size=0.1, random_state=42,
                                                                                    sample_frac=sample_frac,by_time=True)
    train_time = pd.to_datetime(train_time)
    train_time = train_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    test_time = pd.to_datetime(test_time)
    test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    train_loader, test_loader = tensor_data.split_to_loader(train_x, test_x, train_y, test_y, batch_size=batch_size)

    test_x = [torch.tensor(df.values, dtype=torch.float32) for df in test_x['dataframe']]
    test_x = torch.stack(test_x)
    # test_x = test_x.cpu()
    test_x = test_x.cuda()
    len_y = float(test_y.size)
else:
    from dataloader import load_dataset_df,DataFrameToTensor
    df_data, data_list = load_dataset_df()
    tensor_data = DataFrameToTensor(dataframe=df_data, data_list=data_list, label_column=['class', 'type'],
                                    window_size=window_size)
    train_x, test_x, train_y, test_y, train_time, test_time = tensor_data.splittest(test_size=0.1, random_state=42,
                                                                                    sample_frac=sample_frac)
    train_time = pd.to_datetime(train_time)
    train_time = train_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    test_time = pd.to_datetime(test_time)
    test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    train_loader, test_loader = tensor_data.split_to_loader(train_x, test_x, train_y, test_y, batch_size=batch_size)

    test_x = [torch.tensor(df.values, dtype=torch.float32) for df in test_x['dataframe']]
    test_x = torch.stack(test_x)
    # test_x = test_x.cpu()
    test_x = test_x.cuda()
    len_y = float(test_y.size)
def set_random_seed(seed = 10,deterministic=False,benchmark=False):
    random.seed(seed)
    np.random.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
import seaborn as sns
def c_accuracy(y_pred,y_true):
    _,predicted_label = torch.max(y_pred,1)
    correct = (predicted_label == y_true).float()
    accur = correct.sum() / len(correct)
    return accur
def auto_save_file(path):
    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '(0).')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
        path = os.path.join(directory + os.sep + file_name)
    return path
def LSTM_():
    epoch = 500
    # batch_size = 16
    # window_size = 49
    # LSTM_NUM = 2
    lstm_layer_num = 1
    Hidden_Size = 384
    learning_rate = 0.142
    tepoch = 0  # The last epoch that worked best
    # sample_frac = 1.0045536770380648  # >1 Proportion of normal samples to abnormal samples
    use_saved_model = False  # Whether to continue training with the previous model

    # df_data, data_list = load_dataset_df()
    # tensor_data = DataFrameToTensor(dataframe=df_data, data_list=data_list, label_column=['class', 'type'],
    #                                 window_size=window_size)
    # train_x, test_x, train_y, test_y, train_time, test_time = tensor_data.splittest(test_size=0.1, random_state=42,
    #                                                                                 sample_frac=sample_frac)
    # train_time = pd.to_datetime(train_time)
    # train_time = train_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    # test_time = pd.to_datetime(test_time)
    # test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    # train_loader, test_loader = tensor_data.split_to_loader(train_x, test_x, train_y, test_y, batch_size=batch_size)
    # test_x = [torch.tensor(df.values, dtype=torch.float32) for df in test_x['dataframe']]
    # test_x = torch.stack(test_x)
    # # test_x = test_x.cpu()
    # test_x = test_x.cuda()
    # len_y = float(test_y.size)
    # data_loader = tensor_data.create_data_loader(batch_size=batch_size)
    target_list = ['normal', "Leak", "hijack", "Misconfiguration"]
    # mydataset = TensorDataset(tensor_data.data_tensor, tensor_data.label_tensor)

    com_loss_func = nn.CrossEntropyLoss()
    com_model = comLSTM(n_features=54, hidden_dim=Hidden_Size, output_size=4,num_layers=lstm_layer_num)
    optimizer = torch.optim.SGD(com_model.parameters(), lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # print(schedule)
    path = "./checkpoints/model/com_cuda_eopch_" + str(tepoch) + ".pkl"
    start = 0
    if use_saved_model:
        start = tepoch+1
        path_checkpoint = "./checkpoints/model_parameter/test/ckpt_best_%s.pth" % (str(tepoch)) # 断点路径
        checkpoint = torch.load(path_checkpoint)
        com_model.load_state_dict(torch.load(path))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    com_model = com_model.cuda()

    from sklearn.metrics import f1_score
    best_f1_score = 0.0
    best_epoch = 0

    for epoch in range(start, epoch):
        lr_schedule.step()
        for step, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            output = com_model(x)
            loss = com_loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 1:
                eval = com_model(test_x)
                pred_y = torch.max(eval, 1)[1].cpu().data.numpy()
                accuracy = float(np.sum(pred_y == test_y)) / len_y
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                #       '| test accuracy: %.2f' % accuracy)
                from sklearn.metrics import classification_report
                unique_classes = np.unique(np.concatenate([test_y, pred_y]))
                print('Number of unique classes:', len(unique_classes))
                target_names = [f'class_{i}' for i in range(len(unique_classes))]
                test_str = classification_report(y_true=test_y, y_pred=pred_y,
                                                    target_names=target_names, zero_division=1)

                temp_f1 = f1_score(y_pred=pred_y, y_true=test_y
                                   , average='macro', zero_division=1)
                print('temp_f1', temp_f1)
                # temp_sum=temp_f1+temp_route_f1
                if (best_f1_score < temp_f1):
                    best_f1_score = temp_f1
                    best_epoch = epoch
                    print('steps:', epoch)
                    print('step-learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                    checkpoint = {
                        "net": com_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'lr_schedule': lr_schedule.state_dict()
                    }
                    if not os.path.isdir("./checkpoints/model_parameter/test"):
                        os.mkdir("./checkpoints/model_parameter/test")
                    torch.save(checkpoint, './checkpoints/model_parameter/test/com_ckpt_best_%s.pth' % (str(epoch)))
                    # if (epoch == tepoch):
                    path = './checkpoints/model/' + "com_cuda_" + "eopch_" + str(epoch) + ".pkl"
                    torch.save(com_model.state_dict(), path)
                    data = {'id': test_time, 'real': test_y, 'predict': pred_y}
                    result = pd.DataFrame(data)
                    result.to_csv('./checkpoints/res/com_' + str(epoch) + 'reslut.csv', sep='\t', index=0)
        print('epoch:', epoch)
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

        with open('./checkpoints/com_test_best_f1sc_epoch.txt', 'a') as f:
            message = "com_model:" + "LSTM:" + 'epoch:' + str(best_epoch) + ' f1_score:' + str(temp_f1) + '\n'
            f.write(message)


    com_model.load_state_dict(torch.load(path))
    # model.eval()
    valid_acc = []
    features = []
    true_labels = []
    pred_labels = []
    mod_res = pd.DataFrame()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            y = y
            output = com_model(x)
            acc = c_accuracy(output.cpu(), y)
            valid_acc.append(acc.item())
            pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            features.extend(output.cpu().numpy())
            true_labels.extend(y.numpy())
            pred_labels.extend(torch.max(output, 1)[1].cpu().numpy())
            temp_df = pd.DataFrame({'pred_': pred_y, 'true': y.numpy()})
            mod_res = pd.concat([mod_res, temp_df], ignore_index=True)
    valid_run_acc = np.average(valid_acc)
    out_f = np.array(features)
    features_list = [x.cpu().numpy().reshape(x.shape[0], -1) for x, y in test_loader]
    features = np.concatenate(features_list, axis=0)
    # features = np.concatenate([x.cpu().numpy().reshape(-1) for x, y in test_loader])
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    time_labels = np.array(test_time)
    label_map = {0: "Normal", 1: "Leak", 2: "Hijack", 3: "Misconfig"}
    labels_pred = mod_res['pred_'].values
    labels_true = mod_res['true'].values

    colors = ['blue', 'orange', 'green', 'red']
    label_names = ['Normal', 'Leak', 'Hijack', 'Misconfig']
    color_map = dict(zip(range(4), colors))
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2,perplexity=25,early_exaggeration=15, learning_rate=20,random_state=42)
    tsne_results = tsne.fit_transform(features)  # 调整形状以匹配t-SNE的输入要求

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(out_f.reshape(out_f.shape[0], -1))

    # df_tsne = pd.DataFrame()
    # df_tsne['x-tsne'] = pca_result[:, 0]
    # df_tsne['y-tsne'] = pca_result[:, 1]
    # df_tsne['label_pred'] = labels_pred
    # df_tsne['label_true'] = labels_true
    data_for_d3 = {

        'tsne_results': tsne_results.tolist(),  # numpy数组需要转换为列表
        'pca_results':pca_result.tolist(),
        'true_labels': true_labels.tolist(),
        'pred_labels': pred_labels.tolist(),
        'time_labels': time_labels.tolist()
    }

    # 将字典转换为JSON字符串
    json_str = json.dumps(data_for_d3)
    with open('./visdata/com_data_3_d3.json', 'w') as f:
        f.write(json_str)


    # 真实标签的可视化
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    for label in range(4):
        indices = true_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('t-SNE-comlstm colored by true labels')
    plt.legend()

    # 预测标签的可视化
    plt.subplot(2, 2, 2)
    for label in range(4):
        indices = pred_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('t-SNE-comlstm colored by predicted labels')
    plt.legend()

    plt.subplot(2, 2, 3)
    for label in range(4):
        indices = true_labels == label
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('PCA-comlstm colored by true labels')
    plt.legend()
    plt.subplot(2, 2, 4)
    for label in range(4):
        indices = pred_labels == label
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('PCA-comlstm colored by predicted labels')
    plt.legend()

    plt.savefig("tsn-pca-comlstm" + str(tepoch) + ".png")

    plt.show()


def Bilstm_train_():
    epoch = 500
    # batch_size = 16
    # window_size = 49
    LSTM_NUM = 2
    lstm_layer_num = 1
    Hidden_Size = 32
    num_head = 8
    learning_rate = 0.0007086095554615498
    tepoch = 0 #The last epoch that worked best
    # sample_frac = 1.0045536770380648 #>1 Proportion of normal samples to abnormal samples
    use_saved_model = False  #Whether to continue training with the previous model
    # df_data,data_list = load_dataset_df()
    # tensor_data = DataFrameToTensor(dataframe=df_data,data_list=data_list,label_column=['class','type'],window_size=window_size)
    # train_x,test_x,train_y,test_y ,train_time,test_time= tensor_data.splittest(test_size=0.1 , random_state=42 , sample_frac = sample_frac)
    # train_time = pd.to_datetime(train_time)
    # train_time = train_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    # test_time = pd.to_datetime(test_time)
    # test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    # train_loader,test_loader = tensor_data.split_to_loader(train_x,test_x,train_y,test_y,batch_size=batch_size)
    #
    # test_x = [torch.tensor(df.values,dtype=torch.float32 ) for df in test_x['dataframe']]
    # test_x = torch.stack(test_x)
    # # test_x = test_x.cpu()
    # test_x = test_x.cuda()
    # len_y = float(test_y.size)

    current = datetime.datetime.now()

    # data_loader = tensor_data.create_data_loader(batch_size=batch_size)
    target_list = ['normal' ,"Leak","hijack","Misconfiguration"]
    # mydataset = TensorDataset(tensor_data.data_tensor, tensor_data.label_tensor)
    com_loss_func = nn.CrossEntropyLoss()
    model = BISA_LSTM(WINDOW_SIZE=window_size,INPUT_SIZE=54,Hidden_SIZE=Hidden_Size,
                      LSTM_layer_NUM=lstm_layer_num,num_heads=num_head)
    com_model = comLSTM(n_features=54,hidden_dim=128,output_size=4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1)
    # print(schedule)
    path = "./checkpoints/model/BI_cuda_eopch_"+str(tepoch)+".pkl"
    start = 0
    if use_saved_model:
        start = tepoch+1
        path_checkpoint = "./checkpoints/model_parameter/test/BI_ckpt_best_%s.pth" % (str(tepoch)) # 断点路径
        # checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(torch.load(path))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    model = model.cuda()
    from sklearn.metrics import f1_score
    best_f1_score = 0.0
    best_epoch = 0

    for epoch in range(start,epoch):
        for step,(x,y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            output  = model(x)
            # output,attn_weights = model(x)
            loss = com_loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%100 == 1:
                eval = model(test_x)
                # eval,attn_weights = model(test_x)
                pred_y = torch.max(eval,1)[1].cpu().data.numpy()
                accuracy = float(np.sum(pred_y == test_y)) / len_y
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                #       '| test accuracy: %.2f' % accuracy)
                from sklearn.metrics import classification_report
                unique_classes = np.unique(np.concatenate([test_y, pred_y]))
                print('Number of unique classes:', len(unique_classes))
                target_names = [f'class_{i}' for i in range(len(unique_classes))]
                test_str = classification_report(y_true=test_y, y_pred=pred_y,
                                                    target_names=target_names, zero_division=1)
                # temp_str = classification_report(y_true=test_y, y_pred=pred_y,
                #                                  target_names=target_list,zero_division=1)
                temp_f1 = f1_score(y_pred=pred_y, y_true=test_y, average='macro',zero_division=1)
                print('temp_f1', temp_f1)
                # temp_sum=temp_f1+temp_route_f1
                if (best_f1_score < temp_f1):
                    best_f1_score = temp_f1
                    best_epoch = epoch
                    tepoch  = epoch
                    print('epoch:', epoch)
                    print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'lr_schedule': lr_schedule.state_dict()
                    }
                    if not os.path.isdir("./checkpoints/model_parameter/test"):
                        os.mkdir("./checkpoints/model_parameter/test")
                    torch.save(checkpoint, './checkpoints/model_parameter/test/BIckpt_best_%s.pth' % (str(epoch)))
                # if (epoch == tepoch):
                    path = './checkpoints/model/' +"BI_cuda_"+"eopch_"+str(epoch) + ".pkl"
                    torch.save(model.state_dict(), path)
                    data = {'id': test_time, 'real': test_y, 'predict': pred_y}
                    result = pd.DataFrame(data)
                    result.to_csv('./checkpoints/res/BI' + str(epoch) + 'reslut.csv', sep='\t', index=0)
        print('epoch:', epoch)
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        with open('./checkpoints/BI_test_best_f1sc_epoch.txt', 'a') as f:
            message = "model:" + "BI_LSTM:" + 'epoch:' + str(best_epoch) + ' f1_score:' + str(temp_f1) + '\n'
            f.write(message)
        lr_schedule.step()
    path = './checkpoints/model/' + "BI_cuda_" + "eopch_" + str(best_epoch) + ".pkl"
    model.load_state_dict(torch.load(path))
    # model.eval()
    valid_acc = []
    features = []
    true_labels = []
    pred_labels = []
    mod_res = pd.DataFrame()

    with torch.no_grad():
        for x,y in test_loader:
            x= x.cuda()
            y = y
            # output, attn_weights = model(x)
            output = model(x)
            acc = c_accuracy(output.cpu(),y)
            valid_acc.append(acc.item())
            pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            features.extend(output.cpu().numpy())
            true_labels.extend(y.numpy())
            pred_labels.extend(torch.max(output, 1)[1].cpu().numpy())
            temp_df = pd.DataFrame({'pred_': pred_y, 'true': y.numpy()})
            mod_res = pd.concat([mod_res, temp_df], ignore_index=True)
    valid_run_acc = np.average(valid_acc)

    out_f = np.array(features)
    features_list = [x.cpu().numpy().reshape(x.shape[0], -1)  for x, y in test_loader]
    features = np.concatenate(features_list, axis=0)
    # features = np.concatenate([x.cpu().numpy().reshape(-1) for x, y in test_loader])
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    time_labels = np.array(test_time)
    label_map = {0: "Normal", 1: "Leak", 2: "Hijack", 3: "Misconfig"}
    labels_pred = mod_res['pred_'].values
    labels_true = mod_res['true'].values

    colors = ['blue', 'orange', 'green', 'red']
    label_names = ['Normal', 'Leak', 'Hijack', 'Misconfig']
    color_map = dict(zip(range(4), colors))

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=25, early_exaggeration=15, learning_rate=20, random_state=42)

    tsne_results = tsne.fit_transform(features)  # 调整形状以匹配t-SNE的输入要求

    pca = PCA(n_components=2,random_state=42)
    pca_result = pca.fit_transform(out_f.reshape(out_f.shape[0], -1))

    data_for_d3 = {

        'tsne_results': tsne_results.tolist(),  # numpy数组需要转换为列表
        'pca_results': pca_result.tolist(),
        'true_labels': true_labels.tolist(),
        'pred_labels': pred_labels.tolist(),
        'time_labels': time_labels.tolist()
    }
    # 将字典转换为JSON字符串
    json_str = json.dumps(data_for_d3)
    with open('./visdata/BI_4data_for_d3.json', 'w') as f:
        f.write(json_str)
    # df_tsne = pd.DataFrame()
    # df_tsne['x-tsne'] = pca_result[:, 0]
    # df_tsne['y-tsne'] = pca_result[:, 1]
    # df_tsne['label_pred'] = labels_pred
    # df_tsne['label_true'] = labels_true
    # 真实标签的可视化
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    for label in range(4):
        indices = true_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('t-SNE_BILstm colored by true labels')
    plt.legend()

    # 预测标签的可视化
    plt.subplot(2, 2, 2)
    for label in range(4):
        indices = pred_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('t-SNE_BILstm colored by predicted labels')
    plt.legend()

    plt.subplot(2,2,3)
    for label in range(4):
        indices = true_labels == label
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('PCA_BILstm colored by true labels')
    plt.legend()
    plt.subplot(2,2,4)
    for label in range(4):
        indices = pred_labels == label
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('PCA_BILstm colored by predicted labels')
    plt.legend()

    plt.savefig("tsn-pca_BILstm"+str(best_epoch)+".png")

    plt.show()


    test_output = model(test_x)
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
    from sklearn.metrics import classification_report
    unique_classes = np.unique(np.concatenate([test_y, pred_y]))
    print('Number of unique classes:', len(unique_classes))
    target_names = [f'class_{i}' for i in range(len(unique_classes))]
    test_report = classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=target_names, zero_division=1)
    test_parameter_path = './checkpoints/test_parameter/BILstm_'   + '1result.txt'
    with open(test_parameter_path, 'a') as f:
        message =  '\tWINDOW_SIZE:' + str(
            window_size) + "\tLSTM_NUM: " + str(
            LSTM_NUM) + '\tLayer num: ' + str(lstm_layer_num) + '\tLR:' + str(
            learning_rate) + '\tBatch_size: ' + str(
            batch_size) + '\tHidden_size: ' + str(
            Hidden_Size) + '\tNormalizer:MinMaxScaler' + '\t epoch:' + str(
            best_epoch) + '\tf1_score:' + str(best_f1_score) + '\n' + '\t time_bins:60s' + '\n' + test_report + '\n\n'
        print(message)
        f.write(message)

    # torch.save(model, './checkpoints/BI_lstm/'+ 'model.pkl')

    print("Finish")


def base_lstm_train_():
    epoch = 500
    # batch_size = 16
    # window_size = 60
    LSTM_NUM = 2
    lstm_layer_num = 1
    Hidden_Size = 353
    learning_rate = 0.014227390613317552
    tepoch = 0 #The last epoch that worked best
    # sample_frac = 1.002966897294959 #>1 Proportion of normal samples to abnormal samples
    use_saved_model = False  #Whether to continue training with the previous model
    # df_data,data_list = load_dataset_df()
    # tensor_data = DataFrameToTensor(dataframe=df_data,data_list=data_list,label_column=['class','type'],window_size=window_size)
    # train_x,test_x,train_y,test_y ,train_time,test_time= tensor_data.splittest(test_size=0.1 , random_state=42 , sample_frac = sample_frac)
    # train_time = pd.to_datetime(train_time)
    # train_time = train_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    # test_time = pd.to_datetime(test_time)
    # test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    # train_loader,test_loader = tensor_data.split_to_loader(train_x,test_x,train_y,test_y,batch_size=batch_size)
    #
    # test_x = [torch.tensor(df.values,dtype=torch.float32 ) for df in test_x['dataframe']]
    # test_x = torch.stack(test_x)
    # # test_x = test_x.cpu()
    # test_x = test_x.cuda()
    # len_y = float(test_y.size)

    current = datetime.datetime.now()

    # data_loader = tensor_data.create_data_loader(batch_size=batch_size)
    target_list = ['normal' ,"Leak","hijack","Misconfiguration"]
    # mydataset = TensorDataset(tensor_data.data_tensor, tensor_data.label_tensor)
    com_loss_func = nn.CrossEntropyLoss()
    model = SA_LSTM(WINDOW_SIZE=window_size,INPUT_SIZE=54,Hidden_SIZE=Hidden_Size,LSTM_layer_NUM=lstm_layer_num)
    com_model = comLSTM(n_features=54,hidden_dim=128,output_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # print(schedule)
    path = "./checkpoints/model/SA_cuda_eopch_"+str(tepoch)+".pkl"
    start = 0
    if use_saved_model:
        start = tepoch+1
        path_checkpoint = "./checkpoints/model_parameter/test/ckpt_best_%s.pth" % (str(tepoch)) # 断点路径
        # checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(torch.load(path))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    model = model.cuda()
    from sklearn.metrics import f1_score
    best_f1_score = 0.0
    best_epoch = 0

    for epoch in range(start,epoch):
        for step,(x,y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            output  = model(x)
            output,attn_weights = model(x)
            loss = com_loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%100 == 1:
                # eval = model(test_x)
                eval,attn_weights = model(test_x)
                pred_y = torch.max(eval,1)[1].cpu().data.numpy()
                accuracy = float(np.sum(pred_y == test_y)) / len_y
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                #       '| test accuracy: %.2f' % accuracy)
                from sklearn.metrics import classification_report
                unique_classes = np.unique(np.concatenate([test_y, pred_y]))
                print('Number of unique classes:', len(unique_classes))
                target_names = [f'class_{i}' for i in range(len(unique_classes))]
                test_str = classification_report(y_true=test_y, y_pred=pred_y,
                                                    target_names=target_names, zero_division=1)
                # temp_str = classification_report(y_true=test_y, y_pred=pred_y,
                #                                  target_names=target_list,zero_division=1)
                temp_f1 = f1_score(y_pred=pred_y, y_true=test_y, average='macro',zero_division=1)
                print('temp_f1', temp_f1)
                # temp_sum=temp_f1+temp_route_f1
                if (best_f1_score < temp_f1):
                    best_f1_score = temp_f1
                    best_epoch = epoch
                    tepoch  = epoch
                    print('epoch:', epoch)
                    print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'lr_schedule': lr_schedule.state_dict()
                    }
                    if not os.path.isdir("./checkpoints/model_parameter/test"):
                        os.mkdir("./checkpoints/model_parameter/test")
                    torch.save(checkpoint, './checkpoints/model_parameter/test/ckpt_best_%s.pth' % (str(epoch)))
                # if (epoch == tepoch):
                    path = './checkpoints/model/' +"SA_cuda_"+"eopch_"+str(epoch) + ".pkl"
                    torch.save(model.state_dict(), path)
                    data = {'id': test_time, 'real': test_y, 'predict': pred_y}
                    result = pd.DataFrame(data)
                    result.to_csv('./checkpoints/res/' + str(epoch) + 'reslut.csv', sep='\t', index=0)
        print('epoch:', epoch)
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        with open('./checkpoints/test_best_f1sc_epoch.txt', 'a') as f:
            message = "model:" + "SALSTM:" + 'epoch:' + str(best_epoch) + ' f1_score:' + str(temp_f1) + '\n'
            f.write(message)
        lr_schedule.step()
    path = './checkpoints/model/' + "SA_cuda_" + "eopch_" + str(best_epoch) + ".pkl"
    model.load_state_dict(torch.load(path))
    # model.eval()
    valid_acc = []
    features = []
    true_labels = []
    pred_labels = []
    mod_res = pd.DataFrame()
    with torch.no_grad():
        for x,y in test_loader:
            x= x.cuda()
            y = y
            output, attn_weights = model(x)
            acc = c_accuracy(output.cpu(),y)
            valid_acc.append(acc.item())
            pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            features.extend(output.cpu().numpy())
            true_labels.extend(y.numpy())
            pred_labels.extend(torch.max(output, 1)[1].cpu().numpy())
            temp_df = pd.DataFrame({'pred_': pred_y, 'true': y.numpy()})
            mod_res = pd.concat([mod_res, temp_df], ignore_index=True)
    valid_run_acc = np.average(valid_acc)

    out_f = np.array(features)
    features_list = [x.cpu().numpy().reshape(x.shape[0], -1) for x, y in test_loader]
    features = np.concatenate(features_list, axis=0)
    # features = np.concatenate([x.cpu().numpy().reshape(-1) for x, y in test_loader])

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    label_map = {0: "Normal", 1: "Leak", 2: "Hijack", 3: "Misconfig"}
    labels_pred = mod_res['pred_'].values
    labels_true = mod_res['true'].values
    time_labels = np.array(test_time)
    colors = ['blue', 'orange', 'green', 'red']
    label_names = ['Normal', 'Leak', 'Hijack', 'Misconfig']
    color_map = dict(zip(range(4), colors))
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2,perplexity=25,early_exaggeration=15, learning_rate=20,random_state=42)
    # tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)  # 调整形状以匹配t-SNE的输入要求

    pca = PCA(n_components=2,random_state=42)
    pca_result = pca.fit_transform(out_f.reshape(out_f.shape[0], -1))

    # df_tsne = pd.DataFrame()
    # df_tsne['x-tsne'] = pca_result[:, 0]
    # df_tsne['y-tsne'] = pca_result[:, 1]
    # df_tsne['label_pred'] = labels_pred
    # df_tsne['label_true'] = labels_true
    data_for_d3 = {

        'tsne_results': tsne_results.tolist(),  # numpy数组需要转换为列表
        'pca_results': pca_result.tolist(),
        'true_labels': true_labels.tolist(),
        'pred_labels': pred_labels.tolist(),
        'time_labels': time_labels.tolist()
    }
    # 将字典转换为JSON字符串
    json_str = json.dumps(data_for_d3)
    with open('./visdata/SA_0data_for_d3.json', 'w') as f:
        f.write(json_str)
    # 真实标签的可视化
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    for label in range(4):
        indices = true_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('t-SNE colored by true labels')
    plt.legend()

    # 预测标签的可视化
    plt.subplot(2, 2, 2)
    for label in range(4):
        indices = pred_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('t-SNE colored by predicted labels')
    plt.legend()

    plt.subplot(2,2,3)
    for label in range(4):
        indices = true_labels == label
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('PCA colored by true labels')
    plt.legend()
    plt.subplot(2,2,4)
    for label in range(4):
        indices = pred_labels == label
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color_map[label], label=label_names[label],
                    alpha=0.5)
    plt.title('PCA colored by predicted labels')
    plt.legend()

    plt.savefig("tsn-pca"+str(best_epoch)+".png")

    plt.show()

    if test_time.shape[0] == mod_res.shape[0]:
        mod_res.insert(0,"time",test_time)
        # mod_res = pd.concat([test_time,mod_res],axis=1)
    else:
        print("do not eq!")

    mod_res.to_csv(f"./checkpoints/pred_res/epoch"+str(epoch)+".csv")

    test_output, attn_weights = model(test_x)
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
    from sklearn.metrics import classification_report
    unique_classes = np.unique(np.concatenate([test_y, pred_y]))
    print('Number of unique classes:', len(unique_classes))
    target_names = [f'class_{i}' for i in range(len(unique_classes))]
    test_report = classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=target_names, zero_division=1)

    test_parameter_path = './checkpoints/test_parameter/'   + '1result.txt'
    with open(test_parameter_path, 'a') as f:
        message =  '\tWINDOW_SIZE:' + str(
            window_size) + "\tLSTM_NUM: " + str(
            LSTM_NUM) + '\tLayer num: ' + str(lstm_layer_num) + '\tLR:' + str(
            learning_rate) + '\tBatch_size: ' + str(
            batch_size) + '\tHidden_size: ' + str(
            Hidden_Size) + '\tNormalizer:MinMaxScaler' + '\t epoch:' + str(
            best_epoch) + '\tf1_score:' + str(best_f1_score) + '\n' + '\t time_bins:60s' + '\n' + test_report + '\n\n'
        print(message)
        f.write(message)

    torch.save(model, './checkpoints/salstm/'+ 'model.pkl')

    print("Finish")


def main():

    set_random_seed(deterministic=True,benchmark=False)
    Bilstm_train_()
    LSTM_()
    base_lstm_train_()

    # base_lstm_train_()



if __name__ == '__main__':
    main()