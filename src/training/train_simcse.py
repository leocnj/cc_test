import argparse
import os

from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.data.simcse_dataset import SimCSECollator, UnsupervisedSimCSEDataset
from src.models.simcse import SimCSEModel
from src.training.simcse_loss import SimCSELoss


class SimCSETrainer(Trainer):
    def __init__(self, *args, temperature: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = SimCSELoss(temperature=temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # We don't have standard "labels" for unsupervised SimCSE.
        # The collator gives us input_ids and attention_mask.
        # The batch size is 2N because we duplicated each sequence.

        # Remove any labels that might have been automatically passed
        inputs.pop("labels", None)

        # Forward pass: shape will be (2*batch_size, hidden_size)
        embeddings = model(**inputs)

        # Compute InfoNCE loss
        loss = self.loss_fn(embeddings)

        return (loss, embeddings) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Pre-train model with unsupervised SimCSE")
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="Base model name"
    )
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF Unsupervised Dataset")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (actual size will be 2x this during forward pass)",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument(
        "--output_dir", type=str, default="models/simcse_bert", help="Output directory"
    )
    args = parser.parse_args()

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Setup dataset & collator
    train_dataset = UnsupervisedSimCSEDataset(dataset_name=args.dataset, dataset_config=args.config)
    data_collator = SimCSECollator(tokenizer=tokenizer, max_length=args.max_length)

    # Setup model
    model = SimCSEModel(model_name_or_path=args.model_name, pooler_type="cls")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # Default SimCSE optimizer settings often include minor weight decay, etc.
        weight_decay=0.01,
        # Since it's unsupervised, we might not have a clean eval set here, so no eval
        save_strategy="epoch",
        logging_steps=100,
        remove_unused_columns=False,  # Important: our dataset returns raw strings, so Trainer shouldn't remove them automatically
    )

    # Init Trainer
    trainer = SimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        temperature=0.05,
    )

    print("Starting SimCSE pre-training...")
    trainer.train()

    # Save the base model only (without MLP head) for Stage 2
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Base fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
