# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch import optim
import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision.models as models
from LSTM import SA_LSTM
from dataloader import DataFrameToTensor,load_dataset_df
from comLSTM import comLSTM
import datetime
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28,64)
        self.fc2 = torch.nn.Linear(64,64)
        self.fc3 = torch.nn.Linear(64,64)
        self.fc4 = torch.nn.Linear(64,10)


    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x),dim=1)
        return x
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("",is_train,transform=to_tensor,download=True)
    return DataLoader(data_set,batch_size=15,shuffle=True)

def evaluate(test_data,net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x,y) in test_data:
            outputs = net.forward(x.view(-1,28*28))
            for i ,outputs in enumerate(outputs):
                if torch.argmax(outputs) == y[i]:
                    n_correct+=1
                n_total+=1
    return n_correct/n_total


# def get_data(train_ds,vaild_ds,bs):
#     return(
#         DataLoader(train_ds,batch_size=bs,shuffle=True),
#         DataLoader(vaild_ds,batch_size=bs*2)
#     )
def fit(steps ,model, loss_func,opt, train_dl,valid_dl):
    for step in range(steps):
        model.train()
        for xb,yb in train_dl:

            loss_batch(model,loss_func,xb,yb,opt)
        model.eval()
        with torch.no_grad():
            losses,nums = zip(
                *[loss_batch(model,loss_func,xb,yb) for xb,yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses,nums)/np.sum(nums))
        print("This step: "+str(step),"valid loss: "+str(val_loss))
def get_model():
    model = Net()
    return model,optim.SGD(model.parameters(),lr = 0.001)

def loss_batch(model,loss_func,xb,yb,opt=None):
    output = model.forward(xb.view(-1,28*28))
    loss = loss_func(output,yb)
    if opt is not None:

        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)
def ser_parameter_requires_grad(model,feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

# def initialize_model(model_name,num_classes, feature_extract,use_pretrained=True):
#     model_ft = None
#     input_size = 0
#     if model_name =="resnet":
#         model_ft = SA_LSTM(pretrained = use_pretrained)
#         ser_parameter_requires_grad(model_ft,feature_extract)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs,4),torch.nn.LogSoftmax(dim=1))
#         input_size = 224

def splitlstm(lists,m):
    res_lis =[]
    for i in range( len(lists)):
        a = lists[i:i+m]
        res_lis.append(a)
    return np.array(res_lis)

def c_accuracy(y_pred,y_true):
    _,predicted_label = torch.max(y_pred,1)
    correct = (predicted_label == y_true).float()
    accur = correct.sum() / len(correct)
    return accur
def set_random_seed(seed = 10,deterministic=False,benchmark=False):
    random.seed(seed)
    np.random.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
def ex_train(trial):
    epoch = 2500
    batch_size = 16
    window_size = 4
    window_size = trial.suggest_int("window_size",2,120)
    LSTM_NUM = 2
    lstm_layer_num = 1
    Hidden_Size = 128
    Hidden_Size = trial.suggest_int("Hidden_Size",32,512,log=True)
    learning_rate = 0.1
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    tepoch = 0  # The last epoch that worked best
    sample_frac = 5  # >1 Proportion of normal samples to abnormal samples
    sample_frac = trial.suggest_float("sample_frac",1,4)
    use_saved_model = False  # Whether to continue training with the previous model
    df_data, data_list = load_dataset_df()
    tensor_data = DataFrameToTensor(dataframe=df_data, data_list=data_list, label_column=['class', 'type'],
                                    window_size=window_size)
    train_x, test_x, train_y, test_y, train_time, test_time = tensor_data.splittest(test_size=0.1, random_state=42,
                                                                                    sample_frac=sample_frac)
    train_time = pd.to_datetime(train_time)
    test_time = pd.to_datetime(test_time)
    test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    train_loader, test_loader = tensor_data.split_to_loader(train_x, test_x, train_y, test_y, batch_size=batch_size)

    test_x = [torch.tensor(df.values, dtype=torch.float32) for df in test_x['dataframe']]
    test_x = torch.stack(test_x)
    test_x = test_x.cuda()
    len_y = float(test_y.size)
    target_list = ['normal', "Leak", "hijack", "Misconfiguration"]
    com_loss_func = nn.CrossEntropyLoss()
    model = SA_LSTM(WINDOW_SIZE=window_size, INPUT_SIZE=54, Hidden_SIZE=Hidden_Size, LSTM_layer_NUM=lstm_layer_num)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    ex_path = "./checkpoints/model/SA_cuda_eopch_" + str(tepoch) + ".pkl"
    start = 0
    if use_saved_model:
        start = tepoch + 1
        path_checkpoint = "./checkpoints/model_parameter/test/ckpt_best_%s.pth" % (str(tepoch))  # 断点路径
        # checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(torch.load(ex_path))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    model = model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    from sklearn.metrics import f1_score
    best_f1_score = 0.0
    best_epoch = 0

    for epoch in range(start, epoch):
        for step, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            model.train()
            # output = model(x)
            output, attn_weights = model(x)
            loss = com_loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 1:

                # eval = model(test_x)
                eval, attn_weights = model(test_x)
                pred_y = torch.max(eval, 1)[1].cpu().data.numpy()
                accuracy = float(np.sum(pred_y == test_y)) / len_y
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                      '| test accuracy: %.2f' % accuracy)
                from sklearn.metrics import classification_report

                temp_str = classification_report(y_true=test_y, y_pred=pred_y,
                                                 target_names=target_list, zero_division=1)
                temp_f1 = f1_score(y_pred=pred_y, y_true=test_y, average='macro', zero_division=1)
                print('temp_f1', temp_f1)
                # temp_sum=temp_f1+temp_route_f1
                if (best_f1_score < temp_f1):
                    best_f1_score = temp_f1
                    best_epoch = epoch
                    print('epoch:', epoch)
                    print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,

                    }
                    if not os.path.isdir("./checkpoints/model_parameter/test"):
                        os.mkdir("./checkpoints/model_parameter/test")
                    torch.save(checkpoint, './checkpoints/model_parameter/test/ckpt_best_%s.pth' % (str(epoch)))
                    # if (epoch == tepoch):
                    path = './checkpoints/model/' + "opt_SA_cuda_" + "eopch_" + str(epoch) + ".pkl"
                    torch.save(model.state_dict(), path)
                    data = {'id': test_time, 'real': test_y, 'predict': pred_y}
                    result = pd.DataFrame(data)
                    result.to_csv('./checkpoints/res/opt_' + str(epoch) + 'reslut.csv', sep='\t', index=0)
        print(temp_str + '\n' + str(temp_f1))
        with open('./checkpoints/test_best_f1sc_epoch.txt', 'a') as f:
            message = "model:" + "SALSTM:" + 'epoch:' + str(best_epoch) + ' f1_score:' + str(temp_f1) + '\n'
            f.write(message)
        model.eval()
        valid_acc = []
        with torch.no_grad():
            for _,(x, y) in enumerate(test_loader):
                x = x.cuda()
                output, attn_weights = model(x)
                acc = c_accuracy(output.cpu(), y)
                valid_acc.append(acc.item())
                pred_y = torch.max(output, 1)[1].cpu().data.numpy()

        valid_run_acc = np.average(valid_acc)
        trial.report(valid_run_acc,epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()




        # path = './checkpoints/model/' + "SA_cuda_" + "eopch_" + str(best_epoch) + ".pkl"
    model.load_state_dict(torch.load(path))
    # model.eval()
    valid_acc = []
    mod_res = pd.DataFrame()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()

            y = y
            output, attn_weights = model(x)
            acc = c_accuracy(output.cpu(), y)
            valid_acc.append(acc.item())
            pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            temp_df = pd.DataFrame({'pred_': pred_y, 'true': y.numpy()})
            mod_res = pd.concat([mod_res, temp_df], ignore_index=True)
    valid_run_acc = np.average(valid_acc)
    if test_time.shape[0] == mod_res.shape[0]:
        # mod_res=mod_res.insert(0,"time",test_time)
        mod_res = pd.concat([test_time, mod_res], axis=1)
    else:
        print("do not eq!")
    mod_res.to_csv(f"./checkpoints/pred_res/epoch" + str(epoch) + "_optuna.csv")
    print(f'Epoch{epoch + 1},Validation Acc :{valid_run_acc:.4f}')
    model.load_state_dict(torch.load(path))
    test_output, attn_weights = model(test_x)
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
    from sklearn.metrics import classification_report
    test_report = classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=target_list, zero_division=1)
    test_parameter_path = './checkpoints/test_parameter/optuna_' + '1result.txt'
    with open(test_parameter_path, 'a') as f:
        message = '\tWINDOW_SIZE:' + str(
            window_size) + "\tLSTM_NUM: " + str(
            LSTM_NUM) + '\tLayer num: ' + str(lstm_layer_num) + '\tLR:' + str(
            learning_rate) + '\tBatch_size: ' + str(
            batch_size) + '\tHidden_size: ' + str(
            Hidden_Size) + '\tNormalizer:MinMaxScaler' + '\t epoch:' + str(
            best_epoch) + '\tf1_score:' + str(best_f1_score) + '\n' + '\t time_bins:60s' + '\n' + test_report + '\n\n'
        print(message)
        f.write(message)

    torch.save(model, './checkpoints/salstm/' + 'model.pkl')

    print("extrain Finish")
    return valid_run_acc
def train_():
    epoch = 2500
    batch_size = 16
    window_size = 4
    LSTM_NUM = 2
    lstm_layer_num = 1
    Hidden_Size = 128
    learning_rate = 0.01
    tepoch = 1849 #The last epoch that worked best
    sample_frac = 5 #>1 Proportion of normal samples to abnormal samples
    use_saved_model = True  #Whether to continue training with the previous model
    df_data,data_list = load_dataset_df()
    tensor_data = DataFrameToTensor(dataframe=df_data,data_list=data_list,label_column=['class','type'],window_size=window_size)
    train_x,test_x,train_y,test_y ,train_time,test_time= tensor_data.splittest(test_size=0.1 , random_state=42 , sample_frac = sample_frac)
    train_time = pd.to_datetime(train_time)
    train_time = train_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    test_time = pd.to_datetime(test_time)
    test_time = test_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    train_loader,test_loader = tensor_data.split_to_loader(train_x,test_x,train_y,test_y,batch_size=batch_size)

    test_x = [torch.tensor(df.values,dtype=torch.float32 ) for df in test_x['dataframe']]
    test_x = torch.stack(test_x)
    # test_x = test_x.cpu()
    test_x = test_x.cuda()
    len_y = float(test_y.size)
    # test_y_df = test_y
    # test_y = torch.from_numpy(test_y)
    # test_y = test_y.cuda()
    # test_y = torch.tensor(np.array(test_y))
    current = datetime.datetime.now()

    # data_loader = tensor_data.create_data_loader(batch_size=batch_size)
    target_list = ['normal' ,"Leak","hijack","Misconfiguration"]

    # mydataset = TensorDataset(tensor_data.data_tensor, tensor_data.label_tensor)
    com_loss_func = nn.CrossEntropyLoss()

    model = SA_LSTM(WINDOW_SIZE=window_size,INPUT_SIZE=54,Hidden_SIZE=Hidden_Size,LSTM_layer_NUM=lstm_layer_num)
    # model = comLSTM(n_features=55,hidden_dim=128,output_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400, 500], gamma=0.1)
    # print(schedule)
    ex_path = "./checkpoints/model/SA_cuda_eopch_"+str(tepoch)+".pkl"
    start = 0
    if use_saved_model:
        start = tepoch+1
        path_checkpoint = "./checkpoints/model_parameter/test/ckpt_best_%s.pth" % (str(tepoch)) # 断点路径
        # checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(torch.load(ex_path))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    model = model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            lr_schedule.step()
            if step%100 == 1:

                # eval = model(test_x)
                eval,attn_weights = model(test_x)
                pred_y = torch.max(eval,1)[1].cpu().data.numpy()
                accuracy = float(np.sum(pred_y == test_y)) / len_y
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                      '| test accuracy: %.2f' % accuracy)
                from sklearn.metrics import classification_report

                temp_str = classification_report(y_true=test_y, y_pred=pred_y,
                                                 target_names=target_list,zero_division=1)
                temp_f1 = f1_score(y_pred=pred_y, y_true=test_y, average='macro',zero_division=1)
                print('temp_f1', temp_f1)
                # temp_sum=temp_f1+temp_route_f1
                if (best_f1_score < temp_f1):
                    best_f1_score = temp_f1
                    best_epoch = epoch
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
        print(temp_str + '\n' + str(temp_f1))
        with open('./checkpoints/test_best_f1sc_epoch.txt', 'a') as f:
            message = "model:" + "SALSTM:" + 'epoch:' + str(best_epoch) + ' f1_score:' + str(temp_f1) + '\n'
            f.write(message)

        # path = './checkpoints/model/' + "SA_cuda_" + "eopch_" + str(best_epoch) + ".pkl"
    model.load_state_dict(torch.load(path))
    # model.eval()
    valid_acc = []
    mod_res = pd.DataFrame()
    import netron
    path1 = "./checkpoints/netron/_lstm.onnx"
    torch.onnx.export(model, test_x[:1], path1, input_names=['input'], output_names=['output'],opset_version=11)
    netron.start(path)


    with torch.no_grad():
        for x,y in test_loader:
            x= x.cuda()

            y = y
            output, attn_weights = model(x)
            acc = c_accuracy(output.cpu(),y)
            valid_acc.append(acc.item())
            pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            temp_df = pd.DataFrame({'pred_': pred_y, 'true': y.numpy()})
            mod_res = pd.concat([mod_res, temp_df], ignore_index=True)
    valid_run_acc = np.average(valid_acc)
    if test_time.shape[0]==mod_res.shape[0]:
        mod_res.insert(0,"time",test_time)
        # mod_res = pd.concat([test_time,mod_res],axis=1)
    else:
        print("do not eq!")
    mod_res.to_csv(f"./checkpoints/pred_res/epoch"+str(epoch)+".csv")
    print(f'Epoch{epoch+1},Validation Acc :{valid_run_acc:.4f}')
    model.load_state_dict(torch.load(path))
    test_output, attn_weights = model(test_x)
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
    from sklearn.metrics import classification_report
    test_report = classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=target_list,zero_division=1)
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

    torch.save(model, './checkpoints/salstm/'  + 'model.pkl')



    print("Finish")


def main():
    storage_name = "sqlite:///optuna.db"
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=20),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3), direction="maximize",
        study_name="BGP_SA_torch", storage=storage_name, load_if_exists=True
    )

    study.optimize(ex_train, n_trials=100, timeout=1200)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = " + str(best_value))
    print("best_params:")
    print(best_params)
    plot_optimization_history(study)
    plot_parallel_coordinate(study)
    plot_contour(study)
    plot_slice(study)
    set_random_seed(deterministic=True,benchmark=False)
    train_()

    # df_split = splitlstm(df_data, 32)
    # list_tensor_data = DataFrameToTensor(dataframe=df_split,label_column=['class','type'])
    # list_tensor_data.convert_to_tensors()
    #
    # loss_func = F.nll_loss
    #
    # train_dl = get_data_loader(is_train=True)
    # valid_dl = get_data_loader(is_train=False)
    # model,opt = get_model()
    # # fit(25,model,loss_func,opt,train_dl,valid_dl)
    #
    # train_data = get_data_loader(is_train=True)
    # test_data = get_data_loader(is_train=False)
    # # net = Net()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device:",device)
    # modelname = "restnet"
    #
    # feature_extract = True
    #
    # # model_df = models.resnet152()
    # model_df = SA_LSTM()
    # print(model_df )
    # # model_ft ,input_size = initialize_model(modelname,4,feature_extract,use_pretrained=True)
    # # model_ft = model_ft.to(device)
    # filename = r'.\\checkpoints'
    # # parame_to_update = model_ft.parameters()
    # print("params to learn: ")
    # if feature_extract:
    #     parame_to_update = []
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad ==True:
    #             parame_to_update.append(param)
    #             print("\t",name)
    # else:
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t",name)

    # net.to(device)
    # print("net:",net)
    # for name ,parameter in net.named_parameters():
    #     print(name,parameter,parameter.size())
    # # print("Initial accuracy:",evaluate(test_data,net))
    # optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    # for epoch in range(1):
    #     for (x,y) in train_data:
    #         x = torch.from_numpy(x).to(device)
    #         y = torch.from_numpy(y).to(device)
    #
    #         net.zero_grad()
    #         output = net.forward(x.view(-1,28*28))
    #         loss = torch.nn.functional.nll_loss(output,y)
    #         loss.backward()
    #         optimizer.step()
    #     print("epoch",epoch,"accuracy:",evaluate(test_data,net))
    #
    # torch.save(net.state_dict(),'model.pkl')

    # for (n,(x,_)) in enumerate(test_data):
    #     if n>3 :
    #         break
    #     predict = torch.argmax(net.forward(x[0].view(-1,28*28)))
    #     plt.figure(n)
    #     plt.imshow(x[0].view(28,28,1))
    #     plt.title("prediction:"+str(int(predict)))
    # plt.show()
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
