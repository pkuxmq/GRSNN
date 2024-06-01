# GRSNN: Graph Reasoning Spiking Neural Networks #

This is the PyTorch implementation of the paper: Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning **(ICML 2024)**. \[[arxiv](https://arxiv.org/abs/2405.16851)\]

## Dependencies and Installation ##

This codebase is based on PyTorch and [TorchDrug]. It supports training and inference
with multiple GPUs or multiple machines.

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

You may install the dependencies via either conda or pip. Generally, the code works
with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Conda ###

```bash
conda install torchdrug pytorch=1.8.2 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install ogb easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install ogb easydict pyyaml
```

## Reproduction ##

To reproduce the results of GRSNN, use the following command. Alternatively, you
may use `--gpus null` to run on a CPU. All the datasets will be automatically
downloaded in the code.

```bash
python script/run.py -c config/knowledge_graph/fb15k237.yaml --gpus [0]
```

We provide the hyperparameters for each experiment in configuration files.
All the configuration files can be found in `config/*/*.yaml`.

For experiments on inductive relation prediction, you need to additionally specify
the split version with `--version v1`.

To run with multiple GPUs or multiple machines, use the following commands

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/knowledge_graph/fb15k237.yaml --gpus [0,1,2,3]
```

```bash
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 script/run.py -c config/knowledge_graph/fb15k237.yaml --gpus [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```

### Visualize Interpretations ###

Once you have models trained on FB15k237, you can visualize the path interpretations
with the following line. Please replace the checkpoint in the .yaml file with your own path.

```bash
python script/visualize.py -c config/knowledge_graph/fb15k237_visualize.yaml
```


Acknowledgement
--------

The code framework is based on the [NBFNet](https://github.com/DeepGraphLearning/NBFNet) repository. The code for surrogate gradients is modified from the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repository.

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
