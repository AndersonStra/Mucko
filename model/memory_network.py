import json
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import hues
import networkx as nx


'''
    读取单元: 通过每个m的注意力大小，来计算上下文信息c(t)
'''
class MemoryRead(nn.Module):
    def __init__(self, query_embed, memory_embed, att_proj):
        super(MemoryRead, self).__init__()
        self.W_att = nn.Linear(query_embed, att_proj)
        self.U_att = nn.Linear(memory_embed, att_proj)
        self.w_att = nn.Linear(att_proj, 1)
        self.W_cat = nn.Linear(query_embed+memory_embed,query_embed)

    '''
        query_graph : fact graph
        memory_graph_list : visual/Semantic graph 的 memory信息
    '''
    def forward(self, query_graph, memory_graph_list):
        # 计算上下文信息 c
        context_vectors = self.cal_attention_from_memory_graph(query_graph, memory_graph_list)  # size=(num_querys, memory_embed)
        # 更新状态信息 h
        concat = torch.cat((context_vectors, query_graph.ndata['h']), dim=1)
        next_query_vectors = F.relu(self.W_cat(concat))  # size=(num_querys, embed_length)
        query_graph.ndata['h'] = next_query_vectors
        return query_graph

    def cal_attention_from_memory_graph(self,query_graph, memory_graph_list):
        query_features = query_graph.ndata['h']  # size=(num_querys, query_embed)  当前t，fact graph中保存的控制信号h
        query_features = self.W_att(query_features)  # size=(num_querys,att_proj)  W*h
        memory_features_list = [g.ndata['h'] for g in memory_graph_list]  # 取出每个m
        memory_features = torch.stack(memory_features_list)
        memory_features_att = self.U_att(memory_features)  # size=(num_querys, num_mem_graph_nodes, att_proj)
        query_features = query_features.unsqueeze(1).repeat(1, memory_graph_list[0].number_of_nodes(), 1)  # shape(num_querys, num_mem_graph_nodes, att_proj)
        att_values = self.w_att(torch.tanh(query_features+memory_features_att)).squeeze()   # size=(num_querys, num_mem_graph_nodes)
        att_values = F.softmax(att_values, dim=-1)
        att_values = att_values.unsqueeze(-1).repeat(1, 1, memory_features.shape[-1])
        context_features = (att_values*memory_features).sum(1)    # size=(num_querys, memory_embed)

        return context_features

'''
    更新单元： 更新Memory中的每个m
'''
class MemoryWrite(nn.Module):
    def __init__(self, memory_size,query_size, relation_size, hidden_size):
        super(MemoryWrite, self).__init__()
        self.W_msg = nn.Linear(memory_size+relation_size, hidden_size)
        self.W_mem = nn.Linear(memory_size, hidden_size)
        self.W_query = nn.Linear(query_size, hidden_size)
        self.W_all = nn.Linear(3*hidden_size, memory_size)

    def forward(self, query_graph, memory_graph_list):
        for i in range(len(memory_graph_list)):  # 对每一个m操作  （每个m对应一个Visual/Semantic graph）
            query_feature = query_graph.ndata['h'][i]  # 取出第i个
            memory_graph = memory_graph_list[i]
            num_memory_graph_nodes = memory_graph.number_of_nodes()
            query_features = query_feature.unsqueeze(0).repeat(num_memory_graph_nodes, 1)
            memory_graph.ndata['q'] = query_features

        memory_graph_batch = dgl.batch(memory_graph_list)
        memory_graph_batch.update_all(message_func=self.message, reduce_func=self.reduce)
        return dgl.unbatch(memory_graph_batch)

    def message(self, edges):
        msg = self.W_msg(torch.cat((edges.src['h'], edges.data['rel']), dim=1))
        return {'msg': msg}

    def reduce(self, nodes):
        neibor_msg = torch.sum(nodes.mailbox['msg'], dim=1)
        new_memory = torch.cat((self.W_query(nodes.data['q']), self.W_mem(nodes.data['h']), neibor_msg), dim=1)
        new_memory = F.relu(self.W_all(new_memory))
        return {'h': new_memory}


class MemoryNetwork(nn.Module):

    """
        query_input_size: fact_gcn_out_dim, 即每个结点经过gcn图内更新后的大小，对应与论文中模块结构的 h (h初始化是有fact得出的)
        memory_size : M = {m,m,m...} 初始化 m = v  每个memory初始化是由结点的embedding得出的 分别对应visual和semantic
        que_size : question embedding的大小  LSTM h 的size
        query_hidden_size :
    """
    def __init__(self, query_input_size, memory_size, que_szie, query_hidden_size, memory_relation_size, memory_hidden_size, mem_read_att_proj, T):
        super(MemoryNetwork, self).__init__()
        self.W_query = nn.Linear(query_input_size + que_szie, query_hidden_size)
        self.read_memory = MemoryRead(query_embed=query_hidden_size, memory_embed=memory_size, att_proj=mem_read_att_proj)
        self.write_memory = MemoryWrite(memory_size=memory_size, query_size=query_hidden_size,relation_size=memory_relation_size, hidden_size=memory_hidden_size)
        self.T=T

    """
        此部分相当于控制单元的操作：控制信号 h 的初始化以及控制信号的更新
        每次处理一个question
        query_graph : fact_graph fact作为query
        memory_graph : visual_graph 或 Semantic_graph
    """
    def forward(self, query_graph, memory_graph, question):
        num_querys = query_graph.number_of_nodes()  # 当前问题中fact graph中结点的个数
        memory_graph_list = [memory_graph] * num_querys  # 每个fact graph中的Node 对应一个 memory graph（Visual/Semantic graph）

        # 将query graph 的每个节点拼上问题的嵌入
        question = question.unsqueeze(0).repeat(num_querys, 1)
        query_feature = torch.cat((query_graph.ndata['h'], question),dim=1)  # 将每个factNode的embedding与question进行拼接
        query_feature = F.relu(self.W_query(query_feature))  # 此部分省略了上下文信息c 以及增加了ReLu函数
        query_graph.ndata['h'] = query_feature   # 截止此步骤，完成了控制信号 h 的初始化，放入了fact graph中的 h 变量

        for t in range(self.T):  # 循环执行t步，更新fact graph的控制信号 h ， 外面取出 h 参数，放入特定位置
            query_graph = self.read_memory(query_graph, memory_graph_list)  # 读取单元： 读取单元，通过计算每个memory的注意力，计算上下文C(t)
                                                                            # 并更新状态信息，返回更新后的graph
            memory_graph_list = self.write_memory(query_graph, memory_graph_list)  # 更新单元：用于更新memory的信息
        return query_graph


class MemoryGate(nn.Module):
    def __init__(self, vis_mem_size, sem_mem_size, node_size, out_size):
        super(MemoryGate, self).__init__()
        self.gate_layer = nn.Linear(vis_mem_size + sem_mem_size + node_size, vis_mem_size + sem_mem_size + node_size)
        self.fuse_layer = nn.Linear(vis_mem_size+sem_mem_size+node_size,out_size)

    def forward(self, fact_graph_batch):
        node_feature = fact_graph_batch.ndata['hh']  # 取出保存的fact graph中Node的编码
        vis_memory = fact_graph_batch.ndata['vis_mem']  # Visual graph 的状态 h(t+1)
        sem_memory = fact_graph_batch.ndata['sem_mem']
        cat = torch.cat((node_feature, vis_memory, sem_memory), dim=1)
        # cat.shape = [784, 900] 此处是结点的Embedding去拼接
        cat = self.gate_layer(cat)
        gate = torch.sigmoid(cat)
        fuse = self.fuse_layer(gate * cat)
        fact_graph_batch.ndata['h'] = fuse
        return fact_graph_batch
