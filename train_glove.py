from gensim.utils import tokenize
from glovpy import GloVe
from glovpy.utils import reusable
from datasets import load_dataset


def main():
    ds = load_dataset(
        "danish-foundation-models/danish-dynaword", split="train", streaming=True
    )

    @reusable
    def stream_dynaword():
        for row in iter(ds):
            yield list(tokenize(row["text"]))

    model = GloVe(vector_size=300, window_size=15, min_count=10, iter=5, memory=16)
    model.train(stream_dynaword())
    model.wv.save("danglove.kv")


if __name__ == "__main__":
    main()
