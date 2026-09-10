from datasets import load_dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers


def load_train_eval_data():
    retrieval = load_dataset("kardosdrur/synthetic-nordic-retrieval", split="danish")
    retrieval = (
        retrieval.rename_column("user_query", "anchor")
        .rename_column("positive_document", "positive")
        .rename_column("hard_negative_document", "negative")
    )
    retrieval_splits = retrieval.train_test_split(test_size=1000)
    retrieval_train_dataset, retrieval_eval_dataset = (
        retrieval_splits["train"],
        retrieval_splits["test"],
    )
    classification = load_dataset(
        "kardosdrur/synthetic-nordic-classification", split="danish"
    )
    classification = (
        classification.rename_column("input_text", "anchor")
        .rename_column("label", "positive")
        .rename_column("misleading_label", "negative")
    )
    classification_splits = classification.train_test_split(test_size=1000)
    classification_train_dataset, classification_eval_dataset = (
        classification_splits["train"],
        classification_splits["test"],
    )
    train_data = DatasetDict(
        {
            "retrieval": retrieval_train_dataset,
            "classification": classification_train_dataset,
        }
    )
    eval_data = DatasetDict(
        {
            "retrieval": retrieval_eval_dataset,
            "classification": classification_eval_dataset,
        }
    )
    return train_data, eval_data


def main():
    model = SentenceTransformer("kardosdrur/handsker-pretrained-300d")
    train_dataset, eval_dataset = load_train_eval_data()
    loss = MultipleNegativesRankingLoss(model)
    run_name = "handsker-300d"
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-1,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(f"models/{run_name}/final")
    model.push_to_hub("kardosdrur/handsker-300d")


if __name__ == "__main__":
    main()
