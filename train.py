# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import codecs
import time
import argparse

import NN.lstm_classifier_minibatch as lstm_classifier_minibatch
import utils.dictionary as Dict
import utils.data_preprocess as DP
import utils.visualise as VIS
iamport matplotlib as plt

import os
import sys
import random

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

def train(train_data, test_data, dev_data, batch_size=100, deviceid=-1, attention=None, models_dir='models'):
    """ training the model one epoch at a time

    :param training_data: the training data (data_preprocess type)
    :param test_data: the test data (data_preprocess type)
    :param dev_data: the development data (data_preprocess type)
    :param batch_size: the size of the minibatches (default 64)
    :param deviceid: the id of the device to be used (default -1, i.e., CPU)
    :param attention: the type of attention ('dot', 'rte', 'None')
    :param models_dir: the directory to store models in
    """
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50
    EPOCH = 100
    BATCH_SIZE = batch_size
    METRIC = 'eucledian'
    ATTENTION = attention

    model = lstm_classifier_minibatch.SiameseSimilarity(embedding_dim=EMBEDDING_DIM,
                                                     hidden_dim=HIDDEN_DIM,
                                                     dict_size=train_data.get_dict_size(),
                                                     #label_size=train_data.get_labels_size(),
                                                     batch_size=BATCH_SIZE,
                                                     metric = METRIC,
                                                     attention_type= ATTENTION,
                                                     deviceid = deviceid)

    if deviceid > -1:
        model.cuda()

    # Uncomment/comment the loss function you want
#    loss_function = lstm_classifier_minibatch.ContrastiveLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    updates = 0
    loss_history = []
    best_dev_pearson = -1.0
    best_epoch = -1

    for epoch in range(EPOCH):
        print('Start epoch: %d' %epoch)
        loss_float = train_epoch(model, train_data, loss_function, optimizer, batch_size, epoch, deviceid)
        loss_history.append(loss_float)

        dev_pearson, dev_mae, dev_rmse, dev_pred = evaluate(model, dev_data, loss_function, 'dev', deviceid)
        test_pearson, test_mae, test_rmse, test_pred = evaluate(model, test_data, loss_function, 'test', deviceid)

        if best_dev_pearson > -1.0:
            print('Best dev pearson: %.2f in epoch %d' % (best_dev_pearson, best_epoch))

        if dev_pearson > best_dev_pearson:
            best_dev_pearson = dev_pearson
            best_epoch = epoch
            if os.path.exists(models_dir):
                if os.listdir(models_dir):
                    cleanup_best_command = "rm " + os.path.join(models_dir, "best_model_minibatch_*.model")
                    os.system(cleanup_best_command)
            else:
                os.mkdir(models_dir)

            print('New Best Dev!!!')
            torch.save(model, os.path.join(models_dir, "best_model_minibatch_" + str(int(test_pearson*10000)) + "_" + str(epoch) + '.model'))
            with open(os.path.join(models_dir, "best_scores.dev"), "w") as oF:
                oF.write('\n'.join([str(d) for d in dev_pred]))
            with open(os.path.join(models_dir, "best_scores.test"), "w") as oF:
                oF.write('\n'.join([str(d) for d in test_pred]))

            updates = 0
        else:
            updates += 1
            if updates >= 50:
                break
            torch.save(model, os.path.join(models_dir, "model_minibatch_" + str(int(test_pearson*10000)) + "_" + str(epoch) + '.model'))

    VIS.show_plot(list(range(epoch+1)),loss_history)

def train_epoch(model, train_data, loss_function, optimizer, batch_size, epoch, deviceid=-1):
    """ training the model during an epoch

    :param train_data: an iterator over the training data.
    :param loss_function: the loss function.
    :param optimizer: the learning optimizer.
    :param batch_size: the size of the batch
    :param epoch: the current epoch count
    :param deviceid: the id of the device to be used (default -1, i.e., CPU)
    :returns: the loss
    """
    #enable training mode
    model.train()

    avg_loss = 0.0
    avg_acc = 0.0
    total = 0
    total_acc = 0.0
    total_loss = 0.0

    for iter, traindata in enumerate(train_data.get_loader(shuf=True, batch_size=batch_size)):
        train_src, _train_src_length, train_trg, _train_trg_length, train_labels, _train_labels_length = traindata
        train_labels = torch.squeeze(train_labels)
        if deviceid > -1:
            train_src = autograd.Variable(train_src.cuda())
            train_trg = autograd.Variable(train_trg.cuda())
            train_labels = train_labels.cuda()
        else:
            train_src = autograd.Variable(train_src)
            train_trg = autograd.Variable(train_trg)

        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(train_src.t(), train_trg.t())
        (output_src, output_trg, sim) = output

        loss = loss_function(sim, autograd.Variable(train_labels.float()))
        loss.backward()
        optimizer.step()

    print("Current loss {}\n".format(loss.item()))
    return loss.item

def evaluate(model, data, loss_function, name ='dev', deviceid=-1):
    """ evaluate a model.

    :param model: the NMT model
    :param data: the data that we want to evaluate
    :param loss_function: the loss_function
    :param vocabulary_to_idx: the dictionary of words to indexes
    :param label_to_idx: the dictionary of labels to indexes
    :param name: the name of the test
    :param deviceid: the id of the device to be used (default -1, i.e., CPU)
    :returns: the divergence from the loss
    """
    model.eval()
    rmse = 0.0
    mae = 0.0
    avg_loss = 0.0
    total = 0

    sim_list = []
    test_label_list = []

    for iter, testdata in enumerate(data.get_loader(shuf=False, batch_size=1)):
        test_src, _test_src_length, test_trg, _test_trg_length, test_label, _test_labels_length = testdata
        test_label = torch.squeeze(test_label)

        model.batch_size = 1

        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        if deviceid > -1:
            test_src = autograd.Variable(test_src.cuda())
            test_trg = autograd.Variable(test_trg.cuda())
            test_label = test_label.cuda()
        else:
            test_src = autograd.Variable(test_src)
            test_trg = autograd.Variable(test_trg)
        prediction = model(test_src.t(), test_trg.t())
        (prediction_src, prediction_trg, sim) = prediction
        loss = loss_function(sim, autograd.Variable(test_label.float()))
        if deviceid > -1:
            sim_list.append(sim[0].data.cpu().numpy()[0])
            rmse += ((sim[0].data - test_label.float())**2).cpu().numpy()
            mae += torch.abs(sim[0].data - test_label.float()).cpu().numpy()
        else:
            sim_list.append(sim[0].cpu().detach().numpy())
            rmse += ((sim[0].data - test_label.float())**2).numpy()
            mae += torch.abs(sim[0].data - test_label.float()).numpy()

        test_label_list.append(test_label.float())
        total += 1
        avg_loss += loss.item()

    VIS.save_plot2(sim_list, test_label_list, os.path.join(os.getcwd(), 'plots', str(round(time.time())) + '.png'))
    VIS.export_values(sim_list, name)
    avg_loss /= total
    rmse = (rmse / total)**0.5
    mae /= total
    pearson = np.corrcoef(sim_list, test_label_list)[1,0] if not np.isnan(np.corrcoef(sim_list, test_label_list)[1,0]) else 0.0

    print(name + ' avg_loss:%g pearson:%g mae:%g rmse:%g' % (avg_loss, pearson, mae, rmse))
    return pearson, mae, rmse, test_label_list


def main():
    ''' read arguments from the command line and initiate the training.
    '''

    parser = argparse.ArgumentParser(description='Train an LSTM sentence-pair classifier.')
    parser.add_argument('-d', '--data-folder', required=True, help='the folder containing the train, test, dev sets.')
    parser.add_argument('-s', '--source-ext', required=False, default='src', help='the extension of the source files.')
    parser.add_argument('-t', '--target-ext', required=False,  default='trg', help='the extension of the target files.')
    parser.add_argument('-b', '--batch-size', required=False, default=64, help='the batch size.')
    parser.add_argument('-a', '--attention-type', required=False, default=None, help='the attention type: \'dot\', \'rte\', \'None\'.')
    parser.add_argument('-m', '--model-folder', required=False, default='models', help='the directory to save the models')
    parser.add_argument('-g', '--gpuid', required=False, default=-1, help='the ID of the GPU to use.')

    args = parser.parse_args()

    source_train_filename = os.path.join(os.path.realpath(args.data_folder), 'train.' + args.source_ext)
    target_train_filename = os.path.join(os.path.realpath(args.data_folder), 'train.' + args.target_ext)
    source_test_filename = os.path.join(os.path.realpath(args.data_folder), 'test.' + args.source_ext)
    target_test_filename = os.path.join(os.path.realpath(args.data_folder), 'test.' + args.target_ext)
    source_dev_filename = os.path.join(os.path.realpath(args.data_folder), 'dev.' + args.source_ext)
    target_dev_filename = os.path.join(os.path.realpath(args.data_folder), 'dev.' + args.target_ext)

    labels_train = os.path.join(os.path.realpath(args.data_folder), 'train.ter')
    labels_test = os.path.join(os.path.realpath(args.data_folder), 'test.ter')
    labels_dev = os.path.join(os.path.realpath(args.data_folder), 'dev.ter')

    data_dict_file = os.path.join(os.path.realpath(args.data_folder), 'data.dict')
    #labels_dict_file = os.path.join(os.path.realpath(args.data_folder), 'ter.dict')
    dictionary_data = Dict.Dictionary(data_dict_file)
    dictionary_data.load_dictionary()
    #dictionary_labels = Dict.Labels(labels_dict_file)
    #dictionary_labels.load_dictionary()

    train_data = DP.DataLD(source_train_filename, target_train_filename, labels_train, dictionary_data) #, dictionary_labels)
    test_data = DP.DataLD(source_test_filename, target_test_filename, labels_test, dictionary_data)#, dictionary_labels)
    dev_data = DP.DataLD(source_dev_filename, target_dev_filename, labels_dev, dictionary_data)#, dictionary_labels)

    deviceid = -1
    if int(args.gpuid) > -1 and torch.cuda.is_available():
        deviceid = int(args.gpuid)
        print('Using GPU ' + str(deviceid))
        torch.cuda.set_device(deviceid)

    train(train_data, test_data, dev_data, int(args.batch_size), deviceid, args.attention_type)


if __name__ == "__main__":
    main()
