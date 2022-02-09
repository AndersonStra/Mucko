import os
import json
import yaml
import argparse
import numpy as np

from math import log
import dgl
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bisect import bisect
from util.vocabulary import Vocabulary
from util.checkpointing import CheckpointManager, load_checkpoint
from model.model import CMGCNnet
from model.v7w_traindataset import V7WTrainDataset
from model.v7w_testdataset import V7WTestDataset
from model.fvqa_traindataset import FvqaTrainDataset
from model.fvqa_testdataset import FvqaTestDataset
from model.okvqa_traindataset import OkvqaTrainDataset
from model.okvqa_testdataset import OkvqaTestDataset


que_types_dict = {"eight": 0, "nine": 0, "four": 0, "six": 0, "two": 0,
                  "other": 0, "one": 0, "five": 0, "ten": 0, "seven": 0, "three": 0}
que_types_res_dict = {"eight": 0, "nine": 0, "four": 0, "six": 0, "two": 0,
                      "other": 0, "one": 0, "five": 0, "ten": 0, "seven": 0, "three": 0}


def train():
    # ============================================================================================
    #                                 (1) Input Arguments
    # ============================================================================================
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpu-workers", type=int, default=4, help="Number of CPU workers for dataloader.")
    # 快照存储的位置
    parser.add_argument("--save-dirpath", default="exp_v7w/testcheckpoints", help="Path of directory to create checkpoint directory and save checkpoints.")
    # 继续训练之前的模型
    parser.add_argument("--load-pthpath", default="", help="To continue training, path to .pth file of saved checkpoint.")
    parser.add_argument("--overfit", action="store_true", help="Whether to validate on val split after every epoch.")
    parser.add_argument("--validate", action="store_true", help="Whether to validate on val split after every epoch.")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=0, help="List of ids of GPUs to use.")
    parser.add_argument("--dataset", default="v7w", help="dataset that model training on")

    # set mannual seed
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    cudnn.benchmark = True
    cudnn.deterministic = True

    args = parser.parse_args()

    # ============================================================================================
    #                                 (2) Input config file
    # ============================================================================================
    if (args.dataset == 'v7w'):
        config_path = '/home/data1/yjgroup/data_zzh/pr_v7w_memory/model/config_v7w.yml'
    elif(args.dataset == 'fvqa'):
        config_path = '/home/data1/yjgroup/data_zzh/pr_v7w_memory/model/config_fvqa.yml'
    elif(args.dataset == 'okvqa'):
        config_path = '/home/data1/yjgroup/data_zzh/pr_okvqa_memory/model/config_okvqa.yml'
    config = yaml.load(open(config_path))

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

    # device = torch.device("cuda:0") if args.gpus != "cpu" else torch.device("cpu")

    # Print config and args.
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))

    # ============================================================================================
    #                                  Setup Dataset, Dataloader
    # ============================================================================================

    if (args.dataset == 'v7w'):
        print('Loading V7WTrainDataset...')
        train_dataset = V7WTrainDataset(config, overfit=args.overfit, in_memory=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['solver']['batch_size'],
                                      num_workers=args.cpu_workers,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        if args.validate:
            print('Loading V7WTestDataset...')
            val_dataset = V7WTestDataset(config, overfit=args.overfit, in_memory=True)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['solver']['batch_size'],
                                        num_workers=args.cpu_workers,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    elif (args.dataset == 'fvqa'):
        print('Loading FVQATrainDataset...')
        train_dataset = FvqaTrainDataset(config, overfit=args.overfit)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['solver']['batch_size'],
                                      num_workers=args.cpu_workers,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        if args.validate:
            print('Loading FVQATestDataset...')
            val_dataset = FvqaTestDataset(config, overfit=args.overfit)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['solver']['batch_size'],
                                        num_workers=args.cpu_workers,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    elif (args.dataset == 'okvqa'):

        if args.validate:
            print('Loading OKVQATestDataset...')
            val_dataset = OkvqaTestDataset(config, overfit=args.overfit, in_memory=True)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['solver']['batch_size'],
                                        num_workers=args.cpu_workers,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    print('Loading glove...')

    glovevocabulary = Vocabulary(config["dataset"]["word_counts_json"], min_count=config["dataset"]["vocab_min_count"])
    glove = np.load(config['dataset']['glove_vec_path'])
    glove = torch.Tensor(glove)

    # ================================================================================================
    #                                   Setup Model & mutil GPUs
    # ================================================================================================
    print('Building Model...')
    model = CMGCNnet(config,
                     que_vocabulary=glovevocabulary,
                     glove=glove,
                     device=device)

    model = model.to(device)

    if -1 not in args.gpu_ids and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    # ================================================================================================
    #                                Setup Before Traing Loop
    # ================================================================================================

    # If loading from checkpoint, adjust start epoch and load parameters.
    if args.load_pthpath == "":
        start_epoch = 0
    else:
       
        start_epoch = int(args.load_pthpath.split("_")[-1][:-4])

        model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        print("Loading resume model from {}...".format(args.load_pthpath))

   
    if args.validate:

        model.eval()
        answers = []
        preds = []
        que_types = []

        for i, batch in enumerate(tqdm(val_dataloader)):
            for que_type in batch['question_type_list']:
                que_types_dict[que_type] = que_types_dict[que_type] + 1

            with torch.no_grad():
                fact_batch_graph = model(batch)

            fact_graphs = dgl.unbatch(fact_batch_graph)

            for i, fact_graph in enumerate(fact_graphs):
                pred = fact_graph.ndata['h'].squeeze()
                preds.append(pred)
                answers.append(batch['facts_answer_id_list'][i])

            que_types = que_types+batch['question_type_list']

        # calculate top@1,top@3
        acc_1 = cal_acc(answers, preds, que_types=que_types)
        print("acc@1={:.2%}  ".format(acc_1))
        torch.cuda.empty_cache()

        cal_type_acc(que_types_dict, que_types_res_dict)

    print('finished !!!')


def cal_type_acc(que_types_dict, que_types_res_dict):
    for qt in list(que_types_dict.keys()):
        acc = que_types_res_dict[qt] / que_types_dict[qt]
        print(qt, acc*100)


def cal_batch_loss(fact_batch_graph, batch, device, pos_weight, neg_weight):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).to(device)

    for i, fact_graph in enumerate(fact_graphs):
        class_weight = torch.FloatTensor([neg_weight, pos_weight])
        pred = fact_graph.ndata['h'].view(1, -1)  # (n,1)
        answer = answers[i].view(1, -1).to(device)
        pred = pred.squeeze()
        answer = answer.squeeze()
        weight = class_weight[answer.long()].to(device)
        loss_fn = torch.nn.BCELoss(weight=weight)
        loss = loss_fn(pred, answer)
        batch_loss = batch_loss + loss

    return batch_loss / len(answers)





def cal_acc(answers, preds, que_types):
    all_num = len(preds)
    acc_num_1 = 0
  

    for i, answer_id in enumerate(answers):
        pred = preds[i]  # (num_nodes)
        try:
            # top@1
            _, idx_1 = torch.topk(pred, k=1)
           
        except RuntimeError:
          
            continue
        else:
            if idx_1.item() == answer_id:
                acc_num_1 = acc_num_1 + 1
                que_types_res_dict[que_types[i]] = que_types_res_dict[que_types[i]]+1
           

    return acc_num_1 / all_num


def collate_fn(batch):
    res = {}
    qid_list = []
    question_list = []
    question_length_list = []
    img_features_list = []
    img_relations_list = []

    fact_num_nodes_list = []
    facts_node_features_list = []
    facts_e1ids_list = []
    facts_e2ids_list = []
    facts_answer_list = []
    facts_answer_id_list = []

    semantic_num_nodes_list = []
    semantic_node_features_list = []
    semantic_e1ids_list = []
    semantic_e2ids_list = []
    semantic_edge_features_list = []
    semantic_num_nodes_list = []

    question_type_list = []

    for item in batch:
        # question
        qid = item['id']
        qid_list.append(qid)

        question = item['question']
        question_list.append(question)

        question_length = item['question_length']
        question_length_list.append(question_length)

        question_type_list.append(item['question_type'])

        # image
        img_features = item['img_features']
        img_features_list.append(img_features)

        img_relations = item['img_relations']
        img_relations_list.append(img_relations)

        # fact
        fact_num_nodes = item['facts_num_nodes']
        fact_num_nodes_list.append(fact_num_nodes)

        facts_node_features = item['facts_node_features']
        facts_node_features_list.append(facts_node_features)

        facts_e1ids = item['facts_e1ids']
        facts_e1ids_list.append(facts_e1ids)

        facts_e2ids = item['facts_e2ids']
        facts_e2ids_list.append(facts_e2ids)

        facts_answer = item['facts_answer']
        facts_answer_list.append(facts_answer)

        facts_answer_id = item['facts_answer_id']
        facts_answer_id_list.append(facts_answer_id)

        # semantic
        semantic_num_nodes = item['semantic_num_nodes']
        semantic_num_nodes_list.append(semantic_num_nodes)

        semantic_node_features = item['semantic_node_features']
        semantic_node_features_list.append(semantic_node_features)

        semantic_e1ids = item['semantic_e1ids']
        semantic_e1ids_list.append(semantic_e1ids)

        semantic_e2ids = item['semantic_e2ids']
        semantic_e2ids_list.append(semantic_e2ids)

        semantic_edge_features = item['semantic_edge_features']
        semantic_edge_features_list.append(semantic_edge_features)

    res['id_list'] = qid_list
    res['question_list'] = question_list
    res['question_length_list'] = question_length_list
    res['features_list'] = img_features_list
    res['img_relations_list'] = img_relations_list
    res['facts_num_nodes_list'] = fact_num_nodes_list
    res['facts_node_features_list'] = facts_node_features_list
    res['facts_e1ids_list'] = facts_e1ids_list
    res['facts_e2ids_list'] = facts_e2ids_list
    res['facts_answer_list'] = facts_answer_list
    res['facts_answer_id_list'] = facts_answer_id_list
    res['semantic_node_features_list'] = semantic_node_features_list
    res['semantic_e1ids_list'] = semantic_e1ids_list
    res['semantic_e2ids_list'] = semantic_e2ids_list
    res['semantic_edge_features_list'] = semantic_edge_features_list
    res['semantic_num_nodes_list'] = semantic_num_nodes_list
    res['question_type_list'] = question_type_list
    return res


if __name__ == "__main__":
    train()
