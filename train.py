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


from bisect import bisect
from util.vocabulary import Vocabulary
from util.checkpointing import CheckpointManager, load_checkpoint
from model.model import CMGCNnet
from model.model_fvqa import CMGCNnet2

from model.v7w_traindataset import V7WTrainDataset
from model.v7w_testdataset import V7WTestDataset
from model.fvqa_traindataset import FvqaTrainDataset
from model.fvqa_testdataset import FvqaTestDataset
from model.okvqa_traindataset import OkvqaTrainDataset
from model.okvqa_testdataset import OkvqaTestDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'



def train():
    # ============================================================================================
    #                                Input Arguments
    # ============================================================================================
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpu-workers", type=int, default=0, help="Number of CPU workers for dataloader.")
    # 快照存储的位置
    parser.add_argument("--save-dirpath", default="", help="Path of directory to create checkpoint directory and save checkpoints.")
    # 继续训练之前的模型
    parser.add_argument("--load-pthpath", default="", help="To continue training, path to .pth file of saved checkpoint.")
    parser.add_argument("--overfit", action="store_true", help="Whether to validate on val split after every epoch.")
    parser.add_argument("--validate", action="store_false", help="Whether to validate on val split after every epoch.")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=0, help="List of ids of GPUs to use.")
    parser.add_argument("--dataset", default="okvqa", help="dataset that model training on")

    # set mannual seed
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    cudnn.benchmark = True
    cudnn.deterministic = True

    args = parser.parse_args()

    # ============================================================================================
    #                                 Input config file
    # ============================================================================================
    if (args.dataset == 'v7w'):
        config_path = 'model/config_v7w.yml'
    elif(args.dataset == 'fvqa'):
        config_path = 'model/config_fvqa.yml'
    elif(args.dataset == 'okvqa'):
        config_path = 'model/config_okvqa.yml'
    config = yaml.load(open(config_path))

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")



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
                                      collate_fn=collate_fn,
                                      )

        if args.validate:
            print('Loading FVQATestDataset...')
            val_dataset = FvqaTestDataset(config, overfit=args.overfit)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['solver']['batch_size'],
                                        num_workers=args.cpu_workers,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    elif (args.dataset == 'okvqa'):
        print('Loading OKVQATrainDataset...')
        train_dataset = OkvqaTrainDataset(config, overfit=args.overfit, in_memory=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['solver']['batch_size'],
                                      num_workers=args.cpu_workers,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        if args.validate:
            print('Loading OKVQATestDataset...')
            val_dataset = OkvqaTestDataset(config, overfit=args.overfit, in_memory=True)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['solver']['batch_size'],
                                        num_workers=args.cpu_workers,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    # glove 相关
    print('Loading glove...')

    glovevocabulary = Vocabulary(config["dataset"]["word_counts_json"], min_count=config["dataset"]["vocab_min_count"])
    glove = np.load(config['dataset']['glove_vec_path'])
    glove = torch.Tensor(glove)

    # ================================================================================================
    #                                   Setup Model & mutil GPUs
    # ================================================================================================
    print('Building Model...')
    if args.dataset=='fvqa':
        model = CMGCNnet2(config,
                        que_vocabulary=glovevocabulary,
                        glove=glove,
                        device=device)
    elif args.dataset=='okvqa':
        model = CMGCNnet(config,
                        que_vocabulary=glovevocabulary,
                        glove=glove,
                        device=device)

    model = model.to(device)

    if -1 not in args.gpu_ids and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)


    iterations = len(train_dataset) // config["solver"]["batch_size"] + 1

    def lr_lambda_fun(current_iteration: int) -> float:
        """Returns a learning rate multiplier.

        Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
        and then gets multiplied by `lr_gamma` every time a milestone is crossed.
        """
        current_epoch = float(current_iteration) / iterations
        if current_epoch <= config["solver"]["warmup_epochs"]:
            alpha = current_epoch / float(config["solver"]["warmup_epochs"])
            return config["solver"]["warmup_factor"] * (1. - alpha) + alpha
        else:
            idx = bisect(config["solver"]["lr_milestones"], current_epoch)
            return pow(config["solver"]["lr_gamma"], idx)

    weight = torch.FloatTensor([0.9, 0.1]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    # optimizer
    optimizer = optim.Adamax(model.parameters(),
                             lr=config["solver"]["initial_lr"])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
    T = iterations * (config["solver"]["num_epochs"] -
                      config["solver"]["warmup_epochs"] + 1)
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, int(T), eta_min=config["solver"]["eta_min"], last_epoch=-1)

    # ================================================================================================
    #                                Setup Before Traing Loop
    # ================================================================================================

    checkpoint_manager = CheckpointManager(model,
                                           optimizer,
                                           args.save_dirpath,
                                           config=config)

    # If loading from checkpoint, adjust start epoch and load parameters.
    if args.load_pthpath == "":
        start_epoch = 0
    else:
        start_epoch = int(args.load_pthpath.split("_")[-1][:-4])+1

        model_state_dict, optimizer_state_dict = load_checkpoint(
            args.load_pthpath)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        print("Loading resume model from {}...".format(args.load_pthpath))

       

    # ================================================================================================
    #                                    Traing Loop
    # ================================================================================================
    # Forever increasing counter keeping track of iterations completed (for tensorboard logging).
    global_iteration_step = start_epoch * iterations

    for epoch in range(start_epoch, config['solver']['num_epochs']):

        print(f"\nTraining for epoch {epoch}:")

        train_answers = []
        train_preds = []

        for i, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            fact_batch_graph = model(batch)
            batch_loss = cal_batch_loss(fact_batch_graph,
                                        batch,
                                        device,
                                        loss_fn=loss_fn)
           
            batch_loss.backward()
            optimizer.step()

            fact_graphs = dgl.unbatch(fact_batch_graph)
            for i, fact_graph in enumerate(fact_graphs):
                train_pred = fact_graph.ndata['h']  # (num_nodes,1)
                train_preds.append(train_pred[:,1])  # [(num_nodes,)]
                train_answers.append(batch['facts_answer_id_list'][i])

            if global_iteration_step <= iterations * config["solver"][
                    "warmup_epochs"]:
                scheduler.step(global_iteration_step)
            else:
                global_iteration_step_in_2 = iterations * config["solver"][
                    "warmup_epochs"] + 1 - global_iteration_step
                scheduler2.step(int(global_iteration_step_in_2))

            global_iteration_step = global_iteration_step + 1
            torch.cuda.empty_cache()

        # --------------------------------------------------------------------------------------------
        #   ON EPOCH END  (checkpointing and validation)
        # --------------------------------------------------------------------------------------------
        checkpoint_manager.step(epoch)
        train_acc_1 = cal_acc(train_answers, train_preds)
        print("trainacc@1={:.2%} ".format(train_acc_1))
  


        # Validate and report automatic metrics.
        if args.validate:
            model.eval()
            answers = []  # [batch_answers,...]
            preds = []  # [batch_preds,...]
            print(f"\nValidation after epoch {epoch}:")
            for i, batch in enumerate(tqdm(val_dataloader)):
                with torch.no_grad():
                    fact_batch_graph = model(batch)
                batch_loss = cal_batch_loss(fact_batch_graph,
                                            batch,
                                            device,
                                            loss_fn=loss_fn)
                # batch_loss = focal_loss(fact_batch_graph, batch, device)
                fact_graphs = dgl.unbatch(fact_batch_graph)
                for i, fact_graph in enumerate(fact_graphs):
                    pred = fact_graph.ndata['h']  # (num_nodes,1)
                    preds.append(pred[:,1])  # [(num_nodes,)]
                    answers.append(batch['facts_answer_id_list'][i])
            acc_1 = cal_acc(answers, preds)
            print("acc@1={:.2%}".format(acc_1))
            

            model.train()
            torch.cuda.empty_cache()
    print('Train finished !!!')
  


def cal_batch_loss(fact_batch_graph, batch, device, loss_fn):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).to(device)

    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h']  # (n,2)
        answer = answers[i].long().to(device)
        loss = loss_fn(pred, answer)
        batch_loss = batch_loss + loss

    return batch_loss / len(answers)


def focal_loss(fact_batch_graph, batch, device, alpha=0.5, gamma=2):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).float().to(device)

    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h'].squeeze()
        target = torch.FloatTensor(answers[i]).to(device).squeeze()
        loss = -1 * alpha * ((1 - pred) ** gamma) * target * torch.log(pred) - (1 - alpha) * (target ** gamma) * (
            1 - pred) * torch.log(1 - pred)
        batch_loss = batch_loss+loss.mean()
    return batch_loss/len(answers)


def cal_acc(answers, preds):
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

    for item in batch:
        # question
        qid = item['id']
        qid_list.append(qid)

        question = item['question']
        question_list.append(question)

        question_length = item['question_length']
        question_length_list.append(question_length)

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
    return res


if __name__ == "__main__":
    train()
