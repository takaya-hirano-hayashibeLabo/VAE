from torch import nn
import torch

def check_in_out_tensor_size(net:nn.Module,tensor_in:torch.Tensor):
    """
    テンソルのサイズを教えてくれる
    """

    tensor_out:torch.Tensor=net(tensor_in)

    print(f"in size : {tensor_in.shape}")
    print(f"out size : {tensor_out.shape}\n")