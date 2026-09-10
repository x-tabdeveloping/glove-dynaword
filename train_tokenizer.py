from datasets import load_dataset
from tokenizers import normalizers, pre_tokenizers, Tokenizer, models, trainers


def main():
    ds = load_dataset(
        "danish-foundation-models/danish-dynaword", split="train", streaming=True
    )
    normalizer = normalizers.BertNormalizer(
        lowercase=True,
        strip_accents=False,
    )

    def batch_iterator(batch_size=5000):
        # Only keep the text column to avoid decoding the rest of the columns unnecessarily
        tok_dataset = ds.select_columns("text")
        for batch in tok_dataset.iter(batch_size):
            batch_text = [normalizer.normalize_str(doc) for doc in batch["text"]]
            yield batch_text

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=30_000, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=int(7.4e6))
    tokenizer.save("tokenizers/dynatoken")


if __name__ == "__main__":
    main()
