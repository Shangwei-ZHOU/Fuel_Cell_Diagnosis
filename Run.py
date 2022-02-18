import os
import re
import json
import numpy as np
import xlrd
import pandas as pd
import scipy.io as scio
import torch
import xlwt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn
from model import Fuel_Cell_Net_40000
from model import Fuel_Cell_Net_16800
from model import Fuel_Cell_Net_10001
import torch.optim as optim
import torchvision.transforms as transforms
from my_dataset import MyDataset
from ConfusionMatrix import ConfusionMatrix
import random

##Diagnostic_variable=1(Unsampled AC voltage response)
##Diagnostic_variable=2(Downsampled AC voltage response)
##Diagnostic_variable=3(Multi-sine voltage response)
Diagnostic_variable=3
##test_random=0(Chronologically split)
##test_random=1(Randomly split)
test_random=0
test_scale=0.15
class_indices={'Drying':0,'Normal':1,'Flooding':2}
json_str=json.dumps(dict((val,key) for key, val in class_indices.items()),indent=4)
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)

#**********************# Drying data
if Diagnostic_variable==1:
    data_drying_root = './data/AC voltage response/In sequential/original/EIS_Voltage_Drying.mat'
    data_drying_dict = scio.loadmat(data_drying_root)
    data_drying = np.transpose(data_drying_dict['EIS_Voltage_Drying'])
elif Diagnostic_variable==2:
    data_drying_root = './data/AC voltage response/In sequential/downsampling/EIS_Voltage_Drying_Down.mat'
    data_drying_dict = scio.loadmat(data_drying_root)
    data_drying = np.transpose(data_drying_dict['EIS_Voltage_Drying_Down'])
else:
    data_drying_root = './data/AC voltage response/Multi-sine/VR_Fixed_Drying.mat'
    data_drying_dict = scio.loadmat(data_drying_root)
    data_drying = np.transpose(data_drying_dict['VR_Fixed_Drying'])
#************************#normal data
if Diagnostic_variable==1:
    data_normal_root = './data/AC voltage response/In sequential/original/EIS_Voltage_Normal.mat'
    data_normal_dict = scio.loadmat(data_normal_root)
    data_normal = np.transpose(data_normal_dict['EIS_Voltage_Normal'])
elif Diagnostic_variable == 2:
    data_normal_root = './data/AC voltage response/In sequential/downsampling/EIS_Voltage_Normal_Down.mat'
    data_normal_dict = scio.loadmat(data_normal_root)
    data_normal = np.transpose(data_normal_dict['EIS_Voltage_Normal_Down'])
else:
    data_normal_root = './data/AC voltage response/Multi-sine/VR_Fixed_Normal.mat'
    data_normal_dict = scio.loadmat(data_normal_root)
    data_normal = np.transpose(data_normal_dict['VR_Fixed_Normal'])
#************************#Flooding data
if Diagnostic_variable==1:
    data_flooding_root = './data/AC voltage response/In sequential/original/EIS_Voltage_Flooding.mat'
    data_flooding_dict = scio.loadmat(data_flooding_root)
    data_flooding = np.transpose(data_flooding_dict['EIS_Voltage_Flooding'])
elif Diagnostic_variable == 2:
    data_flooding_root = './data/AC voltage response/In sequential/downsampling/EIS_Voltage_Flooding_Down.mat'
    data_flooding_dict = scio.loadmat(data_flooding_root)
    data_flooding = np.transpose(data_flooding_dict['EIS_Voltage_Flooding_Down'])
else:
    data_flooding_root = './data/AC voltage response/Multi-sine/VR_Fixed_Flooding.mat'
    data_flooding_dict = scio.loadmat(data_flooding_root)
    data_flooding = np.transpose(data_flooding_dict['VR_Fixed_Flooding'])

number_dring=data_drying.shape[0]
label_drying=np.ones(data_drying.shape[0])*0
number_normal=data_normal.shape[0]
label_normal=np.ones(number_dring)*1
number_flooding=data_flooding.shape[0]
label_flooding=np.ones(number_flooding)*2
## train-test split
if test_random==1:
    train_range_drying = random.sample(list(range(0, number_dring)), round(number_dring * (1 - test_scale)))
    train_range_normal = random.sample(list(range(0, number_normal)), round(number_normal * (1 - test_scale)))
    train_range_flooding = random.sample(list(range(0, number_flooding)), round(number_flooding * (1 - test_scale)))
    test_range_drying=[x for x in list(range(0, number_dring)) if x not in train_range_drying]
    test_range_normal=[x for x in list(range(0, number_normal)) if x not in train_range_normal]
    test_range_flooding=[x for x in list(range(0, number_flooding)) if x not in train_range_flooding]
    train_data = np.vstack((np.vstack((data_drying[train_range_drying, :],data_normal[train_range_normal, :])),data_flooding[train_range_flooding, :]))
    train_label = np.hstack((np.hstack((label_drying[train_range_drying],label_normal[train_range_normal])),label_flooding[train_range_flooding]))
    test_data = np.vstack((np.vstack((data_drying[test_range_drying, :],data_normal[test_range_normal, :])),data_flooding[test_range_flooding, :]))
    test_label = np.hstack((np.hstack((label_drying[test_range_drying],label_normal[test_range_normal])),label_flooding[test_range_flooding]))
else:
    train_data = np.vstack((np.vstack((data_drying[0:round(number_dring * (1 - test_scale)), :],
                                       data_normal[0:round(number_normal * (1 - test_scale)), :])),
                            data_flooding[0:round(number_flooding * (1 - test_scale)), :]))
    train_label = np.hstack((np.hstack((label_drying[0:round(number_dring * (1 - test_scale))],
                                        label_normal[0:round(number_normal * (1 - test_scale))])),
                             label_flooding[0:round(number_flooding * (1 - test_scale))]))
    test_data = np.vstack((np.vstack((data_drying[round(number_dring * (1 - test_scale)):, :],
                                      data_normal[round(number_normal * (1 - test_scale)):, :])),
                           data_flooding[round(number_flooding * (1 - test_scale)):, :]))
    test_label = np.hstack((np.hstack((label_drying[round(number_dring * (1 - test_scale)):],
                                       label_normal[round(number_normal * (1 - test_scale)):])),
                            label_flooding[round(number_flooding * (1 - test_scale)):]))

train_data=train_data[...,np.newaxis,np.newaxis]
test_data=test_data[...,np.newaxis,np.newaxis]
excel_book = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet_CNN = excel_book.add_sheet('CNN', cell_overwrite_ok=True)

def main():
    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = MyDataset(train_data, train_label,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True, num_workers=0)

    test_dataset = MyDataset(test_data, test_label, transform=transform)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_data.shape[0],
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    if Diagnostic_variable == 1:
        net = Fuel_Cell_Net_40000()
    elif Diagnostic_variable == 2:
        net = Fuel_Cell_Net_16800()
    else:
        net = Fuel_Cell_Net_10001()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    y_epoch = []
    loss_epoch = []
    for epoch in range(100):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            optimizer.zero_grad()
            inputs_fuel_cell=torch.squeeze(inputs,dim=3)
            inputs_fuel_cell=torch.tensor(inputs_fuel_cell,dtype=torch.float32)
            labels=torch.tensor(labels,dtype=torch.int64)
            outputs = net(inputs_fuel_cell)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 5 == 4:
                with torch.no_grad():
                    val_data = torch.squeeze(val_image, dim=3)
                    val_data = torch.tensor(val_data, dtype=torch.float32)
                    val_label = torch.tensor(val_label, dtype=torch.int64)
                    outputs = net(val_data)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == val_label).sum().item() / val_label.size(0)
                    y_epoch.append(accuracy)
                    loss_epoch.append(loss)
                    sheet_CNN.write(len(y_epoch) - 1, 1 + 3 * 1, accuracy*100)
                    sheet_CNN.write(len(loss_epoch) - 1, 2 + 3 * 1, loss.item())
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    if Diagnostic_variable == 1:
        excel_book.save('./result_40000.xls')
    elif Diagnostic_variable == 2:
        excel_book.save('./result_16800.xls')
    else:
        excel_book.save('./result_10001.xls')
    plt.plot(np.arange(0, len(y_epoch)), y_epoch, np.arange(0, len(y_epoch)), loss_epoch)
    plt.show()
    print('Finished Training')
    save_path = './Fuel_Cell_Net.pth'
    torch.save(net.state_dict(), save_path)
    predict_model = predict_y.numpy()
    print(predict_model.shape)
    label_true = val_label.numpy()
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=3, labels=labels)
    confusion.update(predict_model, label_true)
    confusion.plot()
    confusion.summary()

if __name__ == '__main__':
    main()
