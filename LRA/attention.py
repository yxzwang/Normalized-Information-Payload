
import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
from deepspeed.ops.sparse_attention import SparsityConfig, BigBirdSparsityConfig


class TesseractSparsityConfig(SparsityConfig):
    def __init__(self, num_heads, block=16, different_layout_per_head=False):
        super().__init__(num_heads, block, different_layout_per_head)

    def make_layout(self, seq_len):
        """
        Arguments:
             seq_len: required: an integer determining the underling sequence length; must be <= max sequence length

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; for dense everything is 1
        """
        if (seq_len % self.block != 0):
            raise ValueError(f'Sequence Length, {seq_len}, needs to be dividable by Block size {self.block}!')

        num_blocks = seq_len // self.block
        layout = TesseractSparsityConfig.get_mask(num_blocks).long() # (num_blocks, num_blocks)
        layout = layout.expand(self.num_heads, *layout.size()) # (num_heads, num_blocks, num_blocks)
        return layout

    @staticmethod
    def get_mask(length):
        def setmap(n):  
            """
            mask1 和mask2是两种不同的传递方式，mask1:2**k x (k+1); 每一行代表这一行能attention到的位置，k个相邻和自己。
            mask1每次attention找周围相邻点
            mask2:k x 2**k x 2**k
            mask2每次attention只找同一维度的attention
            map:每个位置转化为相应二进制数的位置
            """
            k = math.ceil(math.log2(n))
            tesseractmap = [0b0, 0b1] 
            tesseractdict = {}  
            tesseractmask_1 = torch.zeros((2 ** k, k + 1), dtype=torch.int32)
            tesseractmask_2 = torch.full((k, 2 ** k, 2 ** k), float("-inf")) 

            k_list = [2 ** i for i in range(k)]  

            while len(tesseractmap) < 2 ** k: 
                for m in reversed(tesseractmap):
                    tesseractmap.append(m)
                for i in range(len(tesseractmap)):
                    tesseractmap[i] = tesseractmap[i] << 1
                    if i >= int(len(tesseractmap) / 2):
                        tesseractmap[i] = tesseractmap[i] | 0b1
   

            for num in tesseractmap:
                tesseractdict[num] = [num ^ i for i in k_list]

            for i in range(2 ** k):
                for j in range(k):
                    p = tesseractmap.index(tesseractdict[tesseractmap[i]][j])
                    if p < n:
                        tesseractmask_1[i, j] = p
                    else:
                        tesseractmask_1[i, j] = i
                tesseractmask_1[i, k] = i

            return tesseractmask_1

        nomask_ids = setmap(length)
        mask = torch.zeros([length, length])
        for i in range(length):
            mask[i][nomask_ids[i].long()] = 1.
        return mask
        
class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.dim = config["transformer_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]
        
        self.sparse_types = ["hypercube", "bigbird", "longformer", "global", "local", "random", "local+random"]

        if self.attn_type in self.sparse_types:
            from collections import namedtuple
            from bert_sparse_self_attention import BertSparseSelfAttention
            

            bigbird_num_random = {1024: 5, 2048: 5, 4096: 6,}[config['max_seq_len']]

            sparsity_dict = {
                "hypercube": TesseractSparsityConfig(config["num_head"], block=16),
                "local+random": BigBirdSparsityConfig(config["num_head"], block=16, num_global_blocks=0, num_random_blocks=5, num_sliding_window_blocks=3),
                "global": BigBirdSparsityConfig(config["num_head"], block=16, num_global_blocks=1, num_random_blocks=0, num_sliding_window_blocks=0),
                "bigbird": BigBirdSparsityConfig(config["num_head"], block=16, num_global_blocks=1, num_random_blocks=4, num_sliding_window_blocks=3),
                "longformer": BigBirdSparsityConfig(config["num_head"], block=16, num_global_blocks=1, num_random_blocks=0, num_sliding_window_blocks=3),
                ###########################################################################
                "random": BigBirdSparsityConfig(config["num_head"], block=16, num_global_blocks=0, num_random_blocks=4, num_sliding_window_blocks=0),
                "er-random": None,
            }
            sparsity_config = sparsity_dict[self.attn_type]
            print(sparsity_config.make_layout(16*16)[0])

            bert_config = namedtuple('BertConfig', ['hidden_size', 'num_attention_heads', 'attention_head_size'])(
                config["transformer_dim"], config["num_head"], config["head_dim"]
            )
            self.attn = BertSparseSelfAttention(bert_config, sparsity_config)

        elif self.attn_type == "softmax":
            self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
 
            self.attn = SoftmaxAttention(config)
        else:
            raise NotImplementedError

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):
        if self.attn_type in self.sparse_types:
            # key_paading_mask mode is "add"
            mask =  (~mask.bool()).int() * -10000.
            attn_out = self.attn(X, mask)

        elif self.attn_type == "softmax":
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, Q, K, V, mask)
            else:
                attn_out = self.attn(Q, K, V, mask)
            attn_out = self.combine_heads(attn_out)
        else:
            raise NotImplementedError

        out = self.ff(attn_out)

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
