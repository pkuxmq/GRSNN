from collections.abc import Sequence

import torch
from torch import nn
from torch import autograd

from torch_scatter import scatter_add

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from . import layer
from . import surrogate
import math
import torch.nn.functional as F


@R.register("model.GRSNN")
class GRSNN(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dim, num_relation=None, symmetric=False,
                 aggregate_func='sum', tau=4., Vth=2., delay_bound=4, 
                 surrogate_function='sigmoid', 
                 self_connection=False, tau_c=None, linear_scale=1., 
                 time_steps=10, temporal_decay_ratio=0.95, relation_weight=False, 
                 boundary_query=True, not_delay=False, m_regularization=0., layer_norm=False, 
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False, 
                 num_beam=10, path_topk=10):
        super(GRSNN, self).__init__()

        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        
        self.time_steps = time_steps
        self.temporal_decay_ratio = temporal_decay_ratio
        if surrogate_function == 'normalize':
            self.surrogate_function = surrogate.Normalize(alpha=Vth)
        else:
            self.surrogate_function = surrogate.Sigmoid(alpha=4.)
        self.Vth = Vth
        self.tau = tau
        self.tau_c = tau_c

        self.m_reg = m_regularization

        self.boundary_query = boundary_query

        self.num_beam = num_beam
        self.path_topk = path_topk

        # shared linear operation for all edge connections of SNNs
        if self_connection:
            if aggregate_func == 'pna':
                self.linear_layer = nn.Linear(hidden_dim * 13, hidden_dim)
            else:
                self.linear_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            if aggregate_func == 'pna':
                self.linear_layer = nn.Linear(hidden_dim * 12, hidden_dim)
            else:
                self.linear_layer = nn.Linear(hidden_dim, hidden_dim)
        if dependent:
            if relation_weight:
                self.relation_embedding = nn.Linear(input_dim, double_relation * input_dim)
            else:
                self.relation_embedding = None
            self.relation_embedding_delay = nn.Linear(input_dim, double_relation * input_dim)
        else:
            if relation_weight:
                self.relation_embedding = nn.Embedding(double_relation, input_dim)
            else:
                self.relation_embedding = None
            self.relation_embedding_delay = nn.Embedding(double_relation, input_dim)

        self.snn_layer = layer.SNNAlphaRelationalConv(hidden_dim, double_relation, 
                self.linear_layer, self.relation_embedding, self.relation_embedding_delay, 
                dependent, tau, Vth, delay_bound, self.surrogate_function, self_connection, 
                tau_c, linear_scale, aggregate_func, not_delay=not_delay, layer_norm=layer_norm)

        if self.boundary_query:
            feature_dim = hidden_dim * (time_steps if concat_hidden else 1) + input_dim
        else:
            feature_dim = hidden_dim

        self.query = nn.Embedding(double_relation, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

        n_scale = (1 - math.pow(self.temporal_decay_ratio, self.time_steps)) / (1 - self.temporal_decay_ratio)
        kernel = torch.logspace(0, (self.time_steps-1)*math.log10(self.temporal_decay_ratio), steps=self.time_steps, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / n_scale
        self.final_kernel = nn.Parameter(kernel, requires_grad=False)

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
        # add self loop
        # convert homogeneous graphs to knowledge graphs with 1 relation
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        return graph

    @utils.cached
    def snn_forward(self, graph, h_index, r_index, return_spike=False, graph_grad=False):
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        if self.boundary_query:
            # original query mean 0, std 1, change to mean Vth/2, std Vth/2
            query = query * (self.Vth / 2) + self.Vth / 2
            boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        else:
            boundary.scatter_add_(0, index.unsqueeze(0), torch.ones_like(query).unsqueeze(0) * self.Vth)
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        # initialize the first time step with boundary current injection
        # neuronal charge
        current = torch.zeros_like(boundary)
        membrane_potential = boundary
        # generate spike
        spike = self.surrogate_function(membrane_potential - self.Vth)
        membrane_potential = membrane_potential * (1 - spike)
        # during training, hidden: [spike train, membrane potential, current, trace], size: NN*B*D*T, NN*B*D, NN*B*D, NN*B*D*T
        # during testing, hidden: [spike train, membrane potential, current], size: NN*B*D*T, NN*B*D, NN*B*D
        if self.training:
            if self.tau_c is None:
                trace = - math.e / self.tau * (1. - 1. / self.tau) * spike
            else:
                trace = - (math.exp(-1. / self.tau) / self.tau - math.exp(-1. / self.tau_c) / self.tau_c) / (1. / self.tau - 1. / self.tau_c) * spike
            
            hidden = [spike.unsqueeze(-1), membrane_potential, current, trace.unsqueeze(-1)]
        else:
            hidden = [spike.unsqueeze(-1), membrane_potential, current]

        for t in range(self.time_steps - 1):
            if graph_grad:
                graph.requires_grad_()
            hidden = self.snn_layer(graph, hidden)

        if return_spike:
            # NN*B*D*T
            return hidden[0]

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat([hidden[0].flatten(2), node_query], dim=-1)
        else:
            spike_encoding = torch.sum(hidden[0] * self.final_kernel, dim=-1)
            if self.boundary_query:
                output = torch.cat([spike_encoding, node_query], dim=-1)
            else:
                output = spike_encoding - node_query

        if self.m_reg > 0. and self.training:
            membrane_potential = hidden[1]
            m_mean = torch.mean(torch.abs(membrane_potential))
            if m_mean > 0.2:
                reg_term = self.m_reg * m_mean
            else:
                reg_term = torch.tensor(0.).to(m_mean.device)
        else:
            reg_term = None

        return {
            "node_feature": output,
            "reg_term": reg_term, 
        }

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None, return_reg=False):
        # h_index, t_index: B * num_neg
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.snn_forward(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.snn_forward(graph, t_index[:, 0], r_index[:, 0])
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        if not return_reg:
            return score.view(shape)
        else:
            return score.view(shape), output["reg_term"]

    def get_spike_rate(self, graph, h_index, t_index, r_index=None):
        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        # NN*B*D*T
        output = self.snn_forward(graph, h_index[:, 0], r_index[:, 0], return_spike=True)
        # B*(NN*D)*T
        output = output.transpose(0, 1).flatten(1, 2)
        return torch.mean(torch.sum(output, dim=0), dim=0).cpu(), output.shape[0]

    def get_path_spikes(self, graph, h_index, t_index, r_index=None, k=2):
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
        else:
            graph = self.as_relational_graph(graph)
            r_index = torch.zeros_like(h_index)

        # NN*B*D*T
        output = self.snn_forward(graph, h_index, r_index, return_spike=True)

        # get nodes with bidirectional k-hop
        dst = torch.tensor([dst for src, dst, rel in graph.edge_list if src == h_index]).to(graph.device)
        fw_index = torch.unique(dst)
        for _ in range(k - 1):
            dst = torch.tensor([dst for src, dst, rel in graph.edge_list if src in fw_index]).to(graph.device)
            fw_index = torch.unique(torch.cat([fw_index, dst], dim=0))

        src = torch.tensor([src for src, dst, rel in graph.edge_list if dst == t_index]).to(graph.device)
        bw_index = torch.unique(src)
        for _ in range(k - 1):
            src = torch.tensor([src for src, dst, rel in graph.edge_list if dst in bw_index]).to(graph.device)
            bw_index = torch.unique(torch.cat([bw_index, src], dim=0))

        #all_index = torch.unique(torch.cat([fw_index, bw_index]))
        intersection = torch.isin(fw_index, bw_index)
        all_index = fw_index[intersection]
        all_index = torch.unique(torch.cat([all_index, h_index, t_index]))

        # N*D*T
        spikes = output[all_index, 0]
        subgraph = graph.node_mask(all_index)

        return all_index, spikes, subgraph

    def visualize(self, graph, h_index, t_index, r_index, path_max_len=4):
        assert h_index.numel() == 1 and h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)

        output = self.snn_forward(graph, h_index, r_index, graph_grad=True)
        feature = output["node_feature"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(0, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_weights = graph.edge_weight
        edge_grads = autograd.grad(score, edge_weights)[0]
        with graph.edge():
            graph.edge_grad = edge_grads
        distances, back_edges = self.beam_search_distance([graph], h_index, t_index, self.num_beam, path_max_len)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, graphs, h_index, t_index, num_beam=10, path_max_len=4):
        num_node = graphs[0].num_node
        input = torch.full((num_node, num_beam), float("-inf"), device=self.device)
        input[h_index, 0] = 0

        distances = []
        back_edges = []
        for i in range(path_max_len):
            graph = graphs[0]
            graph = graph.edge_mask(graph.edge_list[:, 0] != t_index)
            node_in, node_out = graph.edge_list.t()[:2]

            message = input[node_in] + graph.edge_grad.unsqueeze(-1)
            msg_source = graph.edge_list.unsqueeze(1).expand(-1, num_beam, -1)

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=self.device) / (num_beam + 1)
            # pick the first occurrence as the previous state
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=num_node)
            msg2out = functional._size_to_index(size[node_out_set] * num_beam)
            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_node)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_node)
            else:
                distance = torch.full((num_node, num_beam), float("-inf"), device=self.device)
                back_edge = torch.zeros(num_node, num_beam, 4, dtype=torch.long, device=self.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths
