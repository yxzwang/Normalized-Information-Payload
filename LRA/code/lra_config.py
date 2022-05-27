
config = {
    "listops":{
        "dataset":{
            "train":96000,
            "dev":2000,
            "test":2000,
        },
        "model":{
            "learn_pos_emb":True,
            "tied_weights":2,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":4,
            "num_layers":2,
            "vocab_size":32,
            "max_seq_len":2000,
            "dropout_prob":0.1,
            "attention_dropout":0.,
            "pooling_mode":"MEAN",
            "num_classes":10,
        },
        "training":{
            "batch_size":32,
            "learning_rate":0.0001,
            "warmup":1000,
            "lr_decay":"cos",
            "weight_decay":0.0001,
            "eval_frequency":50,
            "num_train_steps":5000,
            "num_eval_steps":62,
        },
        "gpu_memory":{
            "softmax":32,
            "global":32,
            "local+random":32,
            "hypercube":32,
            "bigbird":32,
            "longformer":32,
        },
        "extra_attn_config":{
            "softmax":{"attention_grad_checkpointing":True},
            "hypercube":{"attention_grad_checkpointing":False},
            "global":{"attention_grad_checkpointing":False},
            "local+random":{"attention_grad_checkpointing":False},
            "bigbird":{"attention_grad_checkpointing":False},
            "longformer":{"attention_grad_checkpointing":False},
        }
    },
    "image":{
        "dataset":{
            "train":45000,
            "dev":5000,
            "test":10000,
        },
        "model":{
            "learn_pos_emb":True,
            "tied_weights":2,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":4,
            "num_layers":2,
            "vocab_size":512,
            "max_seq_len":1024,
            "dropout_prob":0.3,
            "attention_dropout":0.0,
            "pooling_mode":"MEAN",
            "num_classes": 10,
        },
        "training":{
            "batch_size":256,
            "learning_rate":0.005,
            "warmup":4000,
            "lr_decay":"cos",
            "weight_decay":0,
            "eval_frequency":175,
            "num_train_steps":35000,
            "num_eval_steps":20,
        },
        "gpu_memory":{
            "softmax":128,
            "hypercube":512,
            "global":512,
            "local+random":512,
            "longformer":512,
            "bigbird":512,
        },
        "extra_attn_config":{
            "softmax":{"attention_grad_checkpointing":True},
            "hypercube":{"attention_grad_checkpointing":False},
            "global":{"attention_grad_checkpointing":False},
            "local+random":{"attention_grad_checkpointing":False},
            "longformer":{"attention_grad_checkpointing":False},
            "bigbird":{"attention_grad_checkpointing":False},
        }
    },
    "pathfinder32":{
        "model":{
            "learn_pos_emb":True,
            "tied_weights":2,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":4, # jax uses 8, luna uses 4
            "num_layers":2,
            "vocab_size":512,
            "max_seq_len":1024,
            "dropout_prob":0.2,
            "attention_dropout":0.,
            "pooling_mode":"MEAN",
            "num_classes": 2,
        },
        "training":{
            "batch_size":256, # jax uses batch size 512, setsp 62500
            "learning_rate":0.001,
            "warmup":4000,
            "lr_decay":"cos",
            "weight_decay":0.0,
            "eval_frequency":312,
            "num_train_steps":62500,
            "num_eval_steps":312,
        },
        "gpu_memory":{
            "softmax":128,
            "hypercube":256,
            "global":256,
            "local+random":256,
            "longformer":256,
            "bigbird":256,
        },
        "extra_attn_config":{
            "softmax":{"attention_grad_checkpointing":True},
            "hypercube":{"attention_grad_checkpointing":False},
            "global":{"attention_grad_checkpointing":False},
            "local+random":{"attention_grad_checkpointing":False},
            "longformer":{"attention_grad_checkpointing":False},
            "bigbird":{"attention_grad_checkpointing":False},
        }
    },
    "retrieval":{
        "dataset":{
            "train":147086,
            "dev":18090,
            "test":17437,
        },
        "model":{
            "learn_pos_emb":True,
            "tied_weights":2,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":4,
            "num_layers":2,
            "vocab_size":512,
            "max_seq_len":4000,
            "dropout_prob":0.1,
            "attention_dropout":0.1,
            "pooling_mode":"MEAN",
            "num_classes": 2,
        },
        "training":{
            "batch_size":32,
            "learning_rate":0.0005,
            "warmup":8000,
            "lr_decay":"cos",
            "weight_decay":0.0001,
            "eval_frequency":300,
            "num_train_steps":20000,
            "num_eval_steps":565,
        },
        "gpu_memory":{
            "softmax":32,
            "hypercube":32,
            "global":32,
            "local+random":32,
            "longformer":32,
            "bigbird":32,
        },
        "extra_attn_config":{
            "softmax":{"attention_grad_checkpointing":True},
            "hypercube":{"attention_grad_checkpointing":False},
            "global":{"attention_grad_checkpointing":False},
            "local+random":{"attention_grad_checkpointing":False},
            "longformer":{"attention_grad_checkpointing":False},
            "bigbird":{"attention_grad_checkpointing":False},
        }
    },
    "text":{
        "dataset":{
            "train":25000,
            "dev":25000,
            "test":25000,
        },
        "model":{
            "learn_pos_emb":True,
            "tied_weights":2,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":4,
            "num_layers":2,
            "vocab_size":512,
            "max_seq_len":4000,
            "dropout_prob":0.3,
            "attention_dropout":0.0,
            "pooling_mode":"MEAN",
            "num_classes": 2,
        },
        "training":{
            "batch_size":32,
            "learning_rate":0.00005,
            "warmup":8000,
            "lr_decay":"cos",
            "weight_decay":0.0001,
            "eval_frequency":500,
            "num_train_steps":20000,
            "num_eval_steps":781,
        },
        "gpu_memory":{
            "softmax":32,
            "hypercube":32,
            "global":32,
            "local+random":32,
            "longformer":32,
            "bigbird":32,
        },
        "extra_attn_config":{
            "softmax":{"attention_grad_checkpointing":True},
            "hypercube":{"attention_grad_checkpointing":False},
            "global":{"attention_grad_checkpointing":False},
            "local+random":{"attention_grad_checkpointing":False},
            "longformer":{"attention_grad_checkpointing":False},
            "bigbird":{"attention_grad_checkpointing":False},
        }
    }
}

config["pathfinder32-curv_baseline"] = config["pathfinder32"]
config["pathfinder32-curv_contour_length_9"] = config["pathfinder32"]
config["pathfinder32-curv_contour_length_14"] = config["pathfinder32"]