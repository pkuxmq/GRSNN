import os
import sys
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from grsnn import dataset, layer, model, task, util

import matplotlib.pyplot as plt


vocab_file = os.path.join(os.path.dirname(__file__), "../data/fb15k237_entity.txt")
vocab_file = os.path.abspath(vocab_file)


def load_vocab(dataset):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = [entity_mapping[t] for t in dataset.entity_vocab]
    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]

    return entity_vocab, relation_vocab


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] != "FB15k237":
        raise ValueError("Visualization is only implemented for FB15k237")

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_vocab, relation_vocab = load_vocab(dataset)
    num_relation = len(relation_vocab)

    #for i in range(500):
    #    h, t, r = solver.test_set[i]
    #    logger.warning("%d triplet: %s, %s, %s" % (i, entity_vocab[h], relation_vocab[r], entity_vocab[t]))
    q_list = [3]
    entity_list = ["england, u.k. (Q21)", "leodis (Q39121)", "west yorkshire, england (Q23083)", "pontefract/comments (Q1009235)"]
    for i in q_list:
        file_name = os.path.join(os.path.dirname(__file__), "../results/%d.txt" % (i))
        h, t, r = solver.test_set[i]
        logger.warning("triplet: %s, %s, %s" % (entity_vocab[h], relation_vocab[r], entity_vocab[t]))
        triplet = torch.as_tensor([[h, t, r]], device=solver.device)
        solver.model.eval()
        with torch.no_grad():
            all_index, spikes, subgraph = solver.model.get_path_spikes(triplet)
        all_index = all_index.cpu().tolist()
        spikes = spikes.cpu()
        with open(file_name, 'w', encoding='utf-8') as log_txt:
            log_txt.write(str(all_index))
            log_txt.write('\n\n\n')
            for e in range(spikes.shape[0]):
                log_txt.write(entity_vocab[all_index[e]])
                log_txt.write('\n\n\n')
                ss = spikes[e]
                #for j in range(ss.shape[0]):
                #    log_txt.write(str(ss[j].tolist()))
                #    log_txt.write('\n')
                log_txt.write(str(ss.tolist()))
                log_txt.write('\n')
                log_txt.write('\n\n\n')
            for src, dst, rel in subgraph.edge_list:
                r_name = relation_vocab[rel % num_relation]
                if rel >= num_relation:
                    r_name += "^{-1}"
                log_txt.write("triplet: %s, %s, %s\n" % (entity_vocab[src], r_name, entity_vocab[dst]))

        for e in range(spikes.shape[0]):
            e_name = entity_vocab[all_index[e]]
            if e_name in entity_list:
                ss = spikes[e].tolist()
                fig, ax = plt.subplots(figsize=(4,6))
                for k in range(len(ss)):
                    spike_times = [j for j, x in enumerate(ss[k]) if x == 1.]
                    ax.vlines(spike_times, k - 0.25, k + 0.25, lw=2.5)
                ax.set_xticks([0,2,4,6,8])
                ax.set_yticks([0,8,16,24])
                ax.set_xlim(-1, 10)
                ax.set_ylim(-1, 32)
                ax.set_xlabel('time', fontdict={'family' : 'Times New Roman', 'size':20})
                ax.set_ylabel('neuron', fontdict={'family' : 'Times New Roman', 'size':20})
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                e_name = e_name.replace('/', '')
                fig_name = os.path.join(os.path.dirname(__file__), "../results/%s.pdf" % (e_name))
                plt.savefig(fig_name, dpi=600, bbox_inches='tight', format='pdf')
