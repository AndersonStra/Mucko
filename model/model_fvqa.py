import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as numpy
from util.dynamic_rnn import DynamicRNN
from model.img_gcn import ImageGCN
from model.semantic_gcn import SemanticGCN

from model.fact_gcn import FactGCN
from model.global_gcn import GlobalGCN

from model.memory_network import MemoryNetwork, MemoryGate

import dgl
import networkx as nx
import numpy as np
import hues

# FVQA
class CMGCNnet2(nn.Module):
    def __init__(self, config, que_vocabulary, glove, device):
        '''
        :param config: 配置参数
        :param que_vocabulary: 字典 word 2 index
        :param glove: (voc_size,embed_size)
        '''
        super(CMGCNnet2, self).__init__()
        self.config = config
        self.device = device
        # 构建question glove嵌入层
        self.que_glove_embed = nn.Embedding(len(que_vocabulary), config['model']['glove_embedding_size'])
        # 读入初始参数
        self.que_glove_embed.weight.data = glove
        # 固定初始参数
        self.que_glove_embed.weight.requires_grad = False

        # 问题嵌入lstm
        self.ques_rnn = nn.LSTM(config['model']['glove_embedding_size'],
                                config['model']['lstm_hidden_size'],
                                config['model']['lstm_num_layers'],
                                batch_first=True,
                                dropout=config['model']['dropout'])
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # question guided visual node attention
        # 视觉图结点的注意力计算
        self.vis_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_proj_img = nn.Linear(
            config['model']['img_feature_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_value = nn.Linear(
            config['model']['node_att_ques_img_proj_dims'], 1)

        # question guided visual relation attention
        # 视觉图边的注意力计算
        self.vis_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_proj_rel = nn.Linear(
            config['model']['vis_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)

        # question guided semantic node attention
        # 语义图的结点的注意力计算
        self.sem_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_proj_sem = nn.Linear(
            config['model']['sem_node_dims'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_value = nn.Linear(
            config['model']['sem_node_att_ques_img_proj_dims'], 1)

        # question guided semantic relation attention
        # 语义图的边的注意力计算
        self.sem_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_proj_rel = nn.Linear(
            config['model']['sem_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)

        # question guided fact node attention
        # 事实图结点的注意力计算
        self.fact_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_proj_node = nn.Linear(
            config['model']['fact_node_dims'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_value = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims'], 1)

        # 模态内图卷积更新结点表示
        # new ： ImageGCN
        # image gcn1
        self.img_gcn1 = ImageGCN(config,

                                 in_dim=config['model']['img_feature_size'],
                                 out_dim=config['model']['image_gcn1_out_dim'],
                                 rel_dim=config['model']['vis_relation_dims'])

        # semantic gcn1
        self.sem_gcn1 = SemanticGCN(config,
                                    in_dim=config['model']['sem_node_dims'],
                                    out_dim=config['model']['semantic_gcn1_out_dim'],
                                    rel_dim=config['model']['sem_relation_dims'])
        # fact gcn1
        self.fact_gcn1 = FactGCN(config,
                                 in_dim=config['model']['fact_node_dims'],
                                 out_dim=config['model']['fact_gcn1_out_dim'])

        # new : MemoryNetwork
        # 视觉图 M_V
        self.visual_memory_network = MemoryNetwork(query_input_size=config['model']['fact_gcn1_out_dim'],
                                                   memory_size=config['model']['image_gcn1_out_dim'],
                                                   que_szie=config['model']['lstm_hidden_size'],
                                                   query_hidden_size=config['model']['visual_memory_query_hidden_size'],
                                                   memory_relation_size=config['model']['vis_relation_dims'],
                                                   memory_hidden_size=config['model']['visual_memory_memory_hidden_size'],
                                                   mem_read_att_proj=config['model']['visual_memory_memory_read_att_size'],
                                                   T=config['model']['memory_step'])

        # 语义图 M_S
        self.semantic_memory_network = MemoryNetwork(query_input_size=config['model']['fact_gcn1_out_dim'],
                                                     memory_size=config['model']['semantic_gcn1_out_dim'],
                                                     que_szie=config['model']['lstm_hidden_size'],
                                                     query_hidden_size=config['model']['semantic_memory_query_hidden_size'],
                                                     memory_relation_size=config['model']['sem_relation_dims'],
                                                     memory_hidden_size=config['model']['semantic_memory_memory_hidden_size'],
                                                     mem_read_att_proj=config['model']['semantic_memory_memory_read_att_size'],
                                                     T=config['model']['memory_step'])
        # gate处理后，得到fact graph中，concept的最终表示 vi
        self.memory_gate = MemoryGate(vis_mem_size=config['model']['visual_memory_query_hidden_size'],
                                      sem_mem_size=config['model']['semantic_memory_query_hidden_size'],
                                      node_size=config['model']['fact_gcn1_out_dim'],
                                      out_size=config['model']['memory_gate_out_dim'])

        self.global_gcn = GlobalGCN(config, in_dim=512, out_dim=512)

        # 对每个concept做二元分类，预测作为答案的概率
        self.mlp = nn.Sequential(
            nn.Linear(config['model']['memory_gate_out_dim'] + config['model']['lstm_hidden_size'], 1024), 
            nn.ReLU(),
            #nn.Dropout(0.34),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, batch):

        # ======================================================================================
        #                                    数据处理
        # ======================================================================================

        batch_size = len(batch['id_list'])

        # image（visual graph）： Node表示object  结点直接的边表示object两两之间的位置关系
        images = batch['features_list']  # [(n,36,2048)]  一共36个Node 每个Node的embedding_size为2048
        images = torch.stack(images).to(self.device)  # (batch,36,2048)

        img_relations = batch['img_relations_list']  # 边
        img_relations = torch.stack(img_relations).to(self.device)  # (batch,36,36,7)

        # question
        questions = batch['question_list']  # [(max_length,)]
        questions = torch.stack(questions).to(self.device)  # (batch,max_length)

        questions_len_list = batch['question_length_list']
        questions_len_list = torch.Tensor(questions_len_list).long().to(self.device)

        # semantic graph
        # batch中，第i个问题的语义图中有几个Node
        # semantic_num_nodes_list[i].shape = (1)
        semantic_num_nodes_list = torch.Tensor(batch['semantic_num_nodes_list']).long().to(self.device)

        # batch中，第i个问题的语义图中，每个Node的Embedding
        # semantic_n_features_list[i].shape = (n, 300)
        semantic_n_features_list = batch['semantic_node_features_list']
        semantic_n_features_list = [features.to(self.device) for features in semantic_n_features_list]

        semantic_e1ids_list = batch['semantic_e1ids_list']
        semantic_e1ids_list = [e1ids.to(self.device) for e1ids in semantic_e1ids_list]

        semantic_e2ids_list = batch['semantic_e2ids_list']
        semantic_e2ids_list = [e2ids.to(self.device) for e2ids in semantic_e2ids_list]

        # batch中，第i个问题第语义图中，有几条边，每条边的embedding
        # semantic_e_features_list[i].shape = (n, 300)
        semantic_e_features_list = batch['semantic_edge_features_list']
        semantic_e_features_list = [features.to(self.device) for features in semantic_e_features_list]

        # fact graph
        # batch中，第i个问题的事实图中有几个Node
        # fact_num_nodes_list[i].shape = (1)
        fact_num_nodes_list = torch.Tensor(batch['facts_num_nodes_list']).long().to(self.device)

        # batch中，第i个问题的事实图中，每个Node的Embedding
        # semantic_n_features_list[i].shape = (n, 300)
        facts_features_list = batch['facts_node_features_list']
        facts_features_list = [features.to(self.device) for features in facts_features_list]

        facts_e1ids_list = batch['facts_e1ids_list']
        facts_e1ids_list = [e1ids.to(self.device) for e1ids in facts_e1ids_list]

        facts_e2ids_list = batch['facts_e2ids_list']
        facts_e2ids_list = [e2ids.to(self.device) for e2ids in facts_e2ids_list]

        # batch中，第i个问题的answer
        facts_answer_list = batch['facts_answer_list']
        facts_answer_list = [answer.to(self.device) for answer in facts_answer_list]

        facts_answer_id_list = torch.Tensor(batch['facts_answer_id_list']).long().to(self.device)

        # 初始化权重

        # ===============================================================================================================
        #                               1. embed questions
        # ===============================================================================================================
        ques_embed = self.que_glove_embed(questions).float()  # shape (batch,max_length,300)
        # 这里用最后一个LSTM单元隐层的输出hn当做句子的表示
        _, (ques_embed, _) = self.ques_rnn(ques_embed, questions_len_list)  # qes_embed shape=(batch,hidden_size)

        # ===============================================================================================================
        #                               2. question guided visual node attention
        #                                  视觉图结点注意力计算
        # ===============================================================================================================
        node_att_proj_ques_embed = self.vis_node_att_proj_ques(ques_embed)  # shape (batch,proj_size) # question的编码
        node_att_proj_img_embed = self.vis_node_att_proj_img(images)  # shape (batch,36,proj_size)  # v的编码
        # repeat 为了和image有相同的维数36
        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1], 1)  # shape(batch,36,proj_size)
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        vis_node_att_values = self.vis_node_att_value(node_att_proj_img_sum_ques).squeeze()  # shape(batch,36)
        vis_node_att_values = F.softmax(vis_node_att_values, dim=-1)  # shape(batch,36)

        # ===============================================================================================================
        #                                3. question guided visual relation attention
        #                                   视觉图边的注意力计算
        # ===============================================================================================================
        rel_att_proj_ques_embed = self.vis_rel_att_proj_ques(ques_embed)  # shape(batch,128)  问题 q的embedding
        rel_att_proj_rel_embed = self.vis_rel_att_proj_rel(img_relations)  # shape(batch,36,36,128)
                                                                           # 每一个边 r_{ji} 的embedding
        # 改变question的维度
        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        vis_rel_att_values = self.vis_rel_att_value(rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)

        sem_node_att_val_list = []
        sem_edge_att_val_list = []
        for i in range(batch_size): # 因为数据格式的原因，需要对每个batch分开操作，最后再进行append到list末尾，得到一个batch的attention
            # ===============================================================================================================
            #                                4 question guided semantic node attention
            #                                   语义图结点注意力的计算
            # ===============================================================================================================
            num_node = semantic_num_nodes_list[i]  # n  第i个问题的语义图中，有几个Node
            sem_node_features = semantic_n_features_list[i]  # (n,300)  第i个问题的语义图中，每个Node的Embedding
            q_embed = ques_embed[i]  # (512) 第i个问题的question的Embedding
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            sem_node_att_proj_ques_embed = self.sem_node_att_proj_ques(q_embed)  # shape (n,p)
            sem_node_att_proj_sem_embed = self.sem_node_att_proj_sem(sem_node_features)  # shape (n,p)
            sem_node_att_proj_sem_sum_ques = torch.tanh(
                sem_node_att_proj_ques_embed + sem_node_att_proj_sem_embed)  # shape (n,p)
            sem_node_att_values = self.sem_node_att_value(sem_node_att_proj_sem_sum_ques)  # shape(n,1)
            sem_node_att_values = F.softmax(sem_node_att_values, dim=0)  # shape(n,1)

            sem_node_att_val_list.append(sem_node_att_values)

            # ===============================================================================================================
            #                                5 question guided semantic relation attention
            #                                   语义图边注意力的计算
            # ===============================================================================================================
            num_edge = semantic_e_features_list[i].shape[0]  # n  第i个问题的语义图中，有几条边Edge r_{ij}
            sem_edge_features = semantic_e_features_list[i]  # (n,300)  每个边的Embedding
            qq_embed = ques_embed[i]  # (512)  问题 q
            qq_embed = qq_embed.unsqueeze(0).repeat(num_edge, 1)  # (n,512)
            sem_rel_att_proj_ques_embed = self.sem_rel_att_proj_ques(qq_embed)  # shape (n,p)
            sem_rel_att_proj_rel_embed = self.sem_rel_att_proj_rel(sem_edge_features)  # shape (n,p)
            sem_rel_att_proj_rel_sum_ques = torch.tanh(
                sem_rel_att_proj_ques_embed + sem_rel_att_proj_rel_embed)  # shape (n,p)
            sem_rel_att_values = self.sem_rel_att_value(sem_rel_att_proj_rel_sum_ques)  # shape(n,1)
            sem_rel_att_values = F.softmax(sem_rel_att_values, dim=0)  # shape(n,1)

            sem_edge_att_val_list.append(sem_rel_att_values)

        # ===============================================================================================================
        #                                6 question guided fact node attention
        # ===============================================================================================================
        fact_node_att_values_list = []
        for i in range(batch_size):
            num_node = fact_num_nodes_list[i]  # n  第i个问题的事实图的结点数
            fact_node_features = facts_features_list[i]  # (n,1024)  每个结点的Embedding，size = 1024
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques(q_embed)  # shape (n,p)
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node(fact_node_features)  # shape (n,p)
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  # shape (n,p)
            fact_node_att_values = self.fact_node_att_value(fact_node_att_proj_node_sum_ques)  # shape(n,1)
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0)  # shape(n,1)
            fact_node_att_values_list.append(fact_node_att_values)

        # ===============================================================================================================
        #                             7 Build Image Graph
        # ===============================================================================================================
        # 建图 36 nodes,36*36 edges
        img_graphs = []
        for i in range(batch_size):
            g = dgl.DGLGraph()
            g = g.to(self.device)
            # add nodes
            g.add_nodes(36) # 0-35 node_idx
            # add node features
            g.ndata['h'] = images[i] # images.shape = (batch, 36, 2048)  得出第i个问题的每个node的Embedding
            g.ndata['att'] = vis_node_att_values[i].unsqueeze(-1)  # 第i个问题的每个node的Attention
            g.ndata['batch'] = torch.full([36, 1], i).to(self.device) # torch.Size([36, 1]) full with i  表示这个结点属于第i个question
            # add edges
            for s in range(36):
                for d in range(36):
                    g.add_edge(s, d)
            # add edge features
            g.edata['rel'] = img_relations[i].view(36 * 36, self.config['model']['vis_relation_dims'])
                                                                                        # (1,36,36,7) to shape(36*36,7)
            g.edata['att'] = vis_rel_att_values[i].view(36 * 36, 1)  # (1,36,36) to shape(36*36,1)
            img_graphs.append(g)
        image_batch_graph = dgl.batch(img_graphs)  # 保存batch中每个问题的graph

        # ===============================================================================================================
        #                                8 Build Semantic Graph
        # ===============================================================================================================
        semantic_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph = graph.to(self.device)
            graph.add_nodes(semantic_num_nodes_list[i])  # 第i个question的语义图Node个数
            graph.add_edges(semantic_e1ids_list[i], semantic_e2ids_list[i])  # 加边，e1到e2
            graph.ndata['h'] = semantic_n_features_list[i]  # 每个node的Embedding
            graph.ndata['att'] = sem_node_att_val_list[i]  # 每个node的Attention
            graph.edata['rel'] = semantic_e_features_list[i]  # shape=(n, 300) 每个边的Embedding
            graph.edata['att'] = sem_edge_att_val_list[i]  # shape=(n,1) n个边的attention
            semantic_graphs.append(graph)
        semantic_batch_graph = dgl.batch(semantic_graphs)

        # ===============================================================================================================
        #                                9. Build Fact Graph
        # ===============================================================================================================
        fact_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph = graph.to(self.device)
            graph.add_nodes(fact_num_nodes_list[i])
            graph.add_edges(facts_e1ids_list[i], facts_e2ids_list[i])
            graph.ndata['h'] = facts_features_list[i]  # 每个结点的Embedding
            graph.ndata['att'] = fact_node_att_values_list[i] # attention of each Node
            graph.ndata['batch'] = torch.full([fact_num_nodes_list[i], 1], i).to(self.device)
            graph.ndata['answer'] = facts_answer_list[i]
            fact_graphs.append(graph)
        fact_batch_graph = dgl.batch(fact_graphs)

        # ===============================================================================================================
        #                                8. Intra GCN
        # ===============================================================================================================
        # (1). 对visual graph做 gcn
        image_batch_graph = self.img_gcn1(image_batch_graph)
        # (2) 对semantic graph做 gcn
        semantic_batch_graph = self.sem_gcn1(semantic_batch_graph)
        # (2) 对 fact graph做 gcn
        fact_batch_graph = self.fact_gcn1(fact_batch_graph)
        fact_batch_graph.ndata['hh'] = fact_batch_graph.ndata['h']  # 将起初的embedding保存

        # ===============================================================================================================
        #                                9. Memory network
        # ===============================================================================================================
        image_graph_list = dgl.unbatch(image_batch_graph)  # 将每个item从batch中拆分出来
        semantic_graph_list = dgl.unbatch(semantic_batch_graph)
        fact_graph_list = dgl.unbatch(fact_batch_graph)
        new_fact_graph_list = []  # 保存新的fact结点表示
        for i, fact_graph in enumerate(fact_graph_list):  # 每次处理一个batch中的一个question
            question = ques_embed[i]  # 取出当前question

            num_fact_nodes = fact_graph.number_of_nodes()  # 当前问题的 factNode数量
            image_graph = image_graph_list[i]  # 当前问题的Visual graph
            semantic_graph = semantic_graph_list[i]  # 当前问题的Semantic Graph

            question = ques_embed[i]
            fact_graph_memory_visual = self.visual_memory_network(fact_graph, image_graph, question)  # 得到 h_{t+1}
            fact_graph_memory_semantic = self.semantic_memory_network(fact_graph, semantic_graph, question) # 得到h_{t+1}
            fact_graph.ndata['vis_mem'] = fact_graph_memory_visual.ndata['h']  # return query graph（fact graph）
            fact_graph.ndata['sem_mem'] = fact_graph_memory_semantic.ndata['h'] # 将两个h放入图数据中
            new_fact_graph_list.append(fact_graph)

        # ===============================================================================================================
        #                                10. gate
        # ===============================================================================================================
        new_fact_batch_graph = dgl.batch(new_fact_graph_list)
        fact_batch_graph = self.memory_gate(new_fact_batch_graph)  # 更新每个结点的表示

        

        # 拼接上 question 信息
        fact_graphs = dgl.unbatch(fact_batch_graph)
        new_fact_graphs = []
        for i, fact_graph in enumerate(fact_graphs):
            num_nodes = fact_graph.number_of_nodes()
            q_embed = ques_embed[i]
            q_embed = q_embed.unsqueeze(0).repeat(num_nodes, 1)
            fact_graph.ndata['h'] = torch.cat([fact_graph.ndata['h'], q_embed], dim=1)  # 拼接了question
            new_fact_graphs.append(fact_graph)
        fact_batch_graph = dgl.batch(new_fact_graphs)
        fact_batch_graph.ndata['h'] = self.mlp(fact_batch_graph.ndata['h'])

        return fact_batch_graph
