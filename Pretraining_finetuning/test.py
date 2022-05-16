# import torch
from datasets import load_dataset

# cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb,wnli
if __name__ == "__main__":
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # data = load_dataset("ccdv/arxiv-summarization")
    data = load_dataset("ccdv/pubmed-summarization")
    print(data)