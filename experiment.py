# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys
import random
import codecs
import time
import argparse
from datetime import datetime
import pickle
import gc

import utils.dictionary as Dict
import utils.data_preprocess as DP
import utils.performance_aggregator as PA

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.profiler

import NN.lstm_classifier_minibatch as lstm_classifier_minibatch
import NN.lstm_handler as lstm_handler
import optuna
from optuna.trial import TrialState

from fvcore.nn import FlopCountAnalysis


class Objective:

    def __init__(self, train_data, test_data, dev_data, models_dir, MOO=True):
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data
        self.models_dir = models_dir
        self.MOO = MOO

    def __call__(self, trial):

        # freeing memory
        gc.collect()
        torch.cuda.empty_cache()

        EMBEDDING_DIM = trial.suggest_int('embedding_dim', 48, 248, 8)
        HIDDEN_DIM = trial.suggest_int('hidden_dim', 48, 248, 8)
        EPOCH = 100
        BATCH_SIZE = trial.suggest_int('batch_size', 8, 248, 8)
        batch_size = BATCH_SIZE
        METRIC = 'eucledian'
        ATTENTION = trial.suggest_categorical(
            "attention", ['dot', 'rte', 'nlpAttDot'])  # None
        DROPOUT = trial.suggest_float('dropout', 0.1, 0.3)
        lr = trial.suggest_float('learning rate', 1e-3, 1e-2)

        loss_function = nn.MSELoss()

        model = lstm_classifier_minibatch.SiameseSimilarity(embedding_dim=EMBEDDING_DIM,
                                                            hidden_dim=HIDDEN_DIM,
                                                            dict_size=self.train_data.get_dict_size(),
                                                            # label_size=train_data.get_labels_size(),
                                                            batch_size=BATCH_SIZE,
                                                            metric=METRIC,
                                                            attention_type=ATTENTION,
                                                            dropout=DROPOUT,
                                                            deviceid=deviceid)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_dev_pearson, best_test_pearson, best_epoch = train(
            model, (self.train_data, self.test_data, self.dev_data), self.models_dir, loss_function, optimizer, EPOCH, batch_size)

        flops = flop_count(model, self.test_data)

        trial.set_user_attr("Epochs to converge", best_epoch+1)
        trial.set_user_attr("test_pearson", best_test_pearson)
        trial.set_user_attr("Flops", flops)

        if self.MOO == True:
            return best_dev_pearson, flops
        else:
            return best_dev_pearson


def flop_count(model, data):

    if deviceid > -1:
        model.cuda()
    model.eval()

    dataloader = data.get_loader(shuf=False, batch_size=1)

    for iter, testdata in enumerate(dataloader):
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
        flops = FlopCountAnalysis(model, (test_src.t(), test_trg.t()))
        flops.set_op_handle("aten::lstm", lstm_handler.lstm_flop_count)
        flops.set_op_handle("aten::embedding", None,
                            "aten::mul", None,
                            "aten::softmax", None,
                            "aten::pairwise_distance", None)

        return flops.total()


def train(model, dataset, models_dir, loss_function, optimizer, EPOCH, batch_size):

    train_data, test_data, dev_data = dataset

    if deviceid > -1:
        model.cuda()

    updates = 0
    best_dev_pearson = -1.0
    best_epoch = -1
    best_test_pearson = -1.0

    for epoch in range(EPOCH):

        loss_float = train_epoch(
            model, train_data, loss_function, optimizer, batch_size, deviceid)
        dev_pearson, dev_mae, dev_rmse, dev_pred = evaluate(
            model, dev_data, loss_function, 'dev', deviceid)
        test_pearson, test_mae, test_rmse, test_pred = evaluate(
            model, test_data, loss_function, 'test', deviceid)

        if best_dev_pearson > -1.0:
            print('Best dev pearson: %.2f in epoch %d' %
                  (best_dev_pearson, best_epoch))
            pass

        if dev_pearson > best_dev_pearson:
            best_dev_pearson = dev_pearson
            best_test_pearson = test_pearson
            best_epoch = epoch
            print('New Best Dev!!!')
            updates = 0
        else:
            updates += 1

            if updates >= 5:
                print("#==============#\n#EARLY STOPPING#\n#==============#")

                break

    return best_dev_pearson, best_test_pearson, best_epoch+1


def train_epoch(model, train_data, loss_function, optimizer, batch_size, deviceid=-1):

    # enable training mode
    model.train()

    dataloader = train_data.get_loader(shuf=True, batch_size=batch_size)

    avg_loss = 0.0
    avg_acc = 0.0
    total = 0
    total_acc = 0.0
    total_loss = 0.0

# run with evaluation
    for iter, traindata in enumerate(dataloader):
        # code_iteration_to_profiler(iter)
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
        total_loss += loss.item()
        total += 1

        avg_loss = total_loss / total
        loss.backward()
        optimizer.step()
    print(f"train avg_loss: {avg_loss}")

    return loss.item()


def evaluate(model, data, loss_function, name='dev', deviceid=-1):
    epoch = 1
    model.eval()
    rmse = 0.0
    mae = 0.0
    avg_loss = 0.0
    total = 0

    sim_list = []
    test_label_list = []

    dataloader = data.get_loader(shuf=True, batch_size=1)

    for iter, testdata in enumerate(dataloader):
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
            sim_list.append(sim[0].data.cpu().detach().numpy())
            rmse += ((sim[0].data.detach() -
                      test_label.float())**2).cpu().numpy()
            mae += torch.abs(sim[0].data.detach() -
                             test_label.float()).cpu().numpy()
        else:
            sim_list.append(sim[0].data.cpu().detach().numpy())
            rmse += ((sim[0].data.detach() -
                      test_label.float())**2).numpy()
            mae += torch.abs(sim[0].data.detach() -
                             test_label.float()).numpy()

        test_label_list.append(test_label.cpu().detach().numpy())
        total += 1
        avg_loss += loss.item()

    avg_loss /= total
    rmse = (rmse / total)**0.5
    mae /= total

    pearson = np.corrcoef(sim_list, test_label_list)[1, 0] if not np.isnan(
        np.corrcoef(sim_list, test_label_list)[1, 0]) else 0.0

    print(name + ' avg_loss:%g pearson:%g mae:%g rmse:%g' %
          (avg_loss, pearson, mae, rmse))
    return pearson, mae, rmse, test_label_list


def env_setup():

    source_train_filename = data_path / ('train.' + source_ext)
    target_train_filename = data_path / ('train.' + target_ext)
    labels_train = data_path / ('train.' + y_ext)

    source_test_filename = data_path / ('test.' + source_ext)
    target_test_filename = data_path / ('test.' + target_ext)
    labels_test = data_path / ('test.' + y_ext)

    source_dev_filename = data_path / ('dev.' + source_ext)
    target_dev_filename = data_path / ('dev.' + target_ext)
    labels_dev = data_path / ('dev.' + y_ext)

    train_data = DP.DataLD(source_train_filename, target_train_filename,
                           labels_train, bpe_model_path, bpe_vocab_size=88500)
    test_data = DP.DataLD(source_test_filename, target_test_filename,
                          labels_test, bpe_model_path, bpe_vocab_size=88500)
    dev_data = DP.DataLD(source_dev_filename, target_dev_filename,
                         labels_dev, bpe_model_path, bpe_vocab_size=88500)

    models_dir = Path('models')
    return (train_data, test_data, dev_data, models_dir)


def main():

    #######################
    # EXPERIMENT SETTINGS
    ######################
    # Data (should be fine this way)
    data_path = Path("data/combined_data/")
    data_path = Path.cwd() / data_path
    bpe_model_path = 'utils/bpe_model.model'
    bpe_vocab_size = 88500

    source_ext = "src"
    target_ext = "mt"
    y_ext = "hter"

    deviceid = 0
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = env_setup()

    experiment_number = seed
    n_trials = 100

    sampler_name = 'TPESampler'
    # sampler_name = 'RandomSampler'
    # sampler_name = 'NSGAIISampler'
    # sampler_name = 'MOTPESampler'



    # Experiment execution
    study_name = str(experiment_number) + "_" + \
        sampler_name + "_" + str(n_trials)
    if sampler_name == 'RandomSampler':
        objective = Objective(*dataset)
        sampler = optuna.samplers.RandomSampler(seed=seed)
        study = optuna.create_study(
            study_name=study_name, sampler=sampler, directions=['maximize', 'minimize'])
    elif sampler_name == "NSGAIISampler":
        objective = Objective(*dataset)
        sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=25)
        study = optuna.create_study(
            study_name=study_name, sampler=sampler, directions=['maximize', 'minimize'])
    elif sampler_name == "MOTPESampler":
        objective = Objective(*dataset)
        sampler = optuna.samplers.MOTPESampler(n_startup_trials=65, seed=seed)
        study = optuna.create_study(
            study_name=study_name, sampler=sampler, directions=['maximize', 'minimize'])
    elif sampler_name == "TPESampler":
        objective = Objective(*dataset, MOO=False)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(
            study_name=study_name, sampler=sampler, direction='maximize')

    study.optimize(objective, n_trials=n_trials,
                   timeout=None, gc_after_trial=True)
    with open("studies/" + study_name + ".pkl", "wb") as handle:
        pickle.dump(study, handle)


if __name__ == "__main__":
    main()
