import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers
from torchdrug.layers import functional
from . import surrogate
import math


# TODO consider optimizing memory costs
# tensor indices only support int64 type, how to save memory?
class get_message_with_delay_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_train, delay, node_in, trace):
        # spike_train size: N*B*D*T, delay size: EN*B*D, node_in size: EN, trace size: N*B*D*T
        delay_ = delay.round()
        T = spike_train.shape[-1]
        mask = (delay_ >= T)
        delay_ = delay_.masked_fill_(mask, 0)
        # t_index size: EN*B*D
        t_index = T - 1 - delay_.to(dtype=torch.int64)
        # n_index: EN*1*1, b_index: 1*B*1, d_index: 1*1*D
        n_index = node_in.unsqueeze(-1).unsqueeze(-1)
        b_index = torch.arange(t_index.shape[1], device=t_index.device, dtype=torch.int64).unsqueeze(0).unsqueeze(-1)
        d_index = torch.arange(t_index.shape[2], device=t_index.device, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        # delay_message size: EN*B*D
        delay_message = spike_train[n_index, b_index, d_index, t_index].masked_fill_(mask, 0.)

        if spike_train.requires_grad:
            # d_trace size: EN*B*D, for gradient calculation
            #d_trace = trace[n_index, b_index, d_index, t_index].masked_fill_(mask, 0.)
            if trace is None:
                d_trace = None
            else:
                d_trace = trace[n_index, b_index, d_index, t_index].masked_fill_(mask, 0.)
            ctx.save_for_backward(d_trace, mask, t_index, node_in)
            ctx.T = T
            ctx.N = spike_train.shape[0]

        return delay_message

    @staticmethod
    def backward(ctx, grad_output):
        grad_spike_train = None
        grad_delay = None
        if ctx.needs_input_grad[0]:
            d_trace = ctx.saved_tensors[0]
            mask = ctx.saved_tensors[1]
            t_index = ctx.saved_tensors[2]
            node_in = ctx.saved_tensors[3]

            grad_output = grad_output.masked_fill_(mask, 0)

            grad_spike_train_ = torch.zeros(*grad_output.shape, ctx.T, device=grad_output.device)
            grad_spike_train_.scatter_add_(-1, t_index.unsqueeze(-1), grad_output.unsqueeze(-1))

            grad_spike_train = scatter_add(grad_spike_train_, node_in, dim=0, dim_size=ctx.N)

            #grad_delay = grad_output * d_trace
            if d_trace is not None:
                grad_delay = grad_output * d_trace

        return grad_spike_train, grad_delay, None, None


# current-based LIF SNN, the kernel in SRM corresponds to the Alpha function
class SNNAlphaRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    def __init__(self, dim, num_relation, linear_layer=None, relation_embedding=None, relation_embedding_delay=None, dependent=True, tau=4., Vth=2., delay_bound=4, surrogate_function=surrogate.Sigmoid(), self_connection=False, tau_c=None, linear_scale=1., aggregate_func='sum', not_delay=False, layer_norm=False):
        super(SNNAlphaRelationalConv, self).__init__()
        self.dim = dim
        self.num_relation = num_relation
        self.linear_scale = linear_scale

        # shared synaptic weight
        self.linear_layer = linear_layer
        # for synaptic weight
        self.relation_embedding = relation_embedding
        # for synaptic delay
        self.relation_embedding_delay = relation_embedding_delay
        self.dependent = dependent
        # if true, linear_layer size: (dim*2, dim); else, linear_layer size: (dim, dim)
        self.self_connection = self_connection

        self.not_delay = not_delay

        self.tau = tau
        # tau for the current model
        if tau_c is None:
            self.tau_c = tau
        else:
            self.tau_c = tau_c
        self.Vth = Vth
        self.surrogate_function = surrogate_function

        self.delay_bound = delay_bound

        self.aggregate_func = aggregate_func

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = nn.LayerNorm(dim)

    def message(self, graph, input):
        # during training, input: [spike train, membrane potential, current, trace], size: NN*B*D*T, NN*B*D, NN*B*D, NN*B*D*T
        # during testing, input: [spike train, membrane potential, current], size: NN*B*D*T, NN*B*D, NN*B*D
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            if self.relation_embedding is not None:
                # consider synaptic weight for relations
                relation_input = self.relation_embedding(graph.query).view(batch_size, self.num_relation, self.dim)
            # synaptic delay
            relation_input_delay = self.relation_embedding_delay(graph.query).view(batch_size, self.num_relation, self.dim)
        else:
            if self.relation_embedding is not None:
                # consider synaptic weight for relations
                relation_input = self.relation_embedding.weight.expand(batch_size, -1, -1)
            # synaptic delay
            relation_input_delay = self.relation_embedding_delay.weight.expand(batch_size, -1, -1)

        if self.relation_embedding is not None:
            relation_input = relation_input.transpose(0, 1)
            edge_input = relation_input[relation]

        # synaptic delay
        relation_input_delay = relation_input_delay.transpose(0, 1)
        edge_input_delay = torch.sigmoid(relation_input_delay[relation]) * self.delay_bound

        spike_train = input[0]
        if self.not_delay:
            message = spike_train[..., -1][node_in]
        else:
            if self.training:
                trace = input[3]
            else:
                trace = None
            message = get_message_with_delay_v2.apply(spike_train, edge_input_delay, node_in, trace)

        if self.relation_embedding is not None:
            message = message * edge_input

        return message

    def aggregate(self, graph, message):
        # message size: EN*B*D
        # node_out size: EN
        node_out = graph.edge_list[:, 1]
        # edge_weight size: EN*1*1
        edge_weight = graph.edge_weight.unsqueeze(-1).unsqueeze(-1)

        # update size: NN*B*D
        # to fulfill the properties of SNNs, we will only consider 'sum' for GRSNN
        # possible for future improvement
        if self.aggregate_func == 'sum':
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == 'max':
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        # input currents of the boundary will be injected into nodes at every time step
        update = [update, graph.boundary]

        return update

    def message_and_aggregate(self, graph, input):
        return super(SNNAlphaRelationalConv, self).message_and_aggregate(graph, input)

    def combine(self, input, update):
        # during training, input: [spike train, membrane potential, current, trace], size: NN*B*D*T, NN*B*D, NN*B*D, NN*B*D*T
        # during testing, input: [spike train, membrane potential, current], size: NN*B*D*T, NN*B*D, NN*B*D
        # update: [update, graph.boundary], size: NN*B*D
        NN = input[0].shape[0]
        current_injection = update[1]
        if self.self_connection:
            linear_input = torch.cat([input[0][..., -1], update[0]], dim=-1)
        else:
            linear_input = update[0]
        update_current = self.linear_layer(linear_input)
        update_current *= self.linear_scale

        if self.layer_norm:
            update_current = self.ln(update_current)

        membrane_potential = input[1]
        current = input[2]
        # neuronal charge
        if self.tau == self.tau_c:
            current = math.exp(-1. / self.tau) * current + update_current * (math.e / self.tau)
        else:
            current = math.exp(-1. / self.tau_c) * current + update_membrane_potential
        membrane_potential = math.exp(-1. / self.tau) * membrane_potential + current + current_injection
        # generate spike
        spike = self.surrogate_function(membrane_potential - self.Vth)
        membrane_potential = membrane_potential * (1 - spike)

        # concat output
        if self.training and not self.not_delay:
            spike_train = input[0]
            spike_train_new = torch.cat([spike_train, spike.unsqueeze(-1)], dim=-1)
            with torch.no_grad():
                if self.tau == self.tau_c:
                    trace_kernel = torch.tensor([math.e / self.tau * (1 - (i + 1) / self.tau) * math.exp(- i / self.tau) for i in range(spike_train.shape[-1], -1, -1)], device=spike_train.device, dtype=torch.float)
                else:
                    trace_kernel = torch.tensor([(math.exp(-i / self.tau) / self.tau - math.exp(-i / self.tau_c) / self.tau_c) / (1. / self.tau - 1. / self.tau_c) for i in range(spike_train.shape[-1] + 1, 0, -1)], device=spike_train.device, dtype=torch.float)
                trace = torch.sum(spike_train_new * trace_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1)
                trace = -trace
                trace.requires_grad_(False)
            output = [spike_train_new, membrane_potential, current, torch.cat([input[3], trace.unsqueeze(-1)], dim=-1)]
        else:
            output = [torch.cat([input[0], spike.unsqueeze(-1)], dim=-1), membrane_potential, current]
        return output

