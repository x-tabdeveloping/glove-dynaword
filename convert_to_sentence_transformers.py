import numpy as np
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.sentence_transformer.modules.tokenizer.whitespace import (
    WhitespaceTokenizer,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

kv = KeyedVectors.load("danglove.kv")
vocab = kv.index_to_key
embeddings = kv.vectors
vocab = ["[UNK]"] + vocab
embeddings = np.concatenate([np.zeros((1, embeddings.shape[1])), embeddings])
key_to_idx = {word: idx for idx, word in enumerate(vocab)}

tokenizer = Tokenizer(WordLevel(vocab=key_to_idx, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

encoder = SentenceTransformer(
    None,
    modules=[
        StaticEmbedding(
            tokenizer,
            embeddings,
        )
    ],
)

encoder.push_to_hub("kardosdrur/handsker-pretrained-300d", exist_ok=True)
