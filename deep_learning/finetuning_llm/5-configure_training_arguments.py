#!/usr/bin/env python3
"""Configure training arguments for fine-tuning a model."""
import transformers


def configure_training_args(output_dir, epochs, per_device_train_batch_size,
                            per_device_eval_batch_size, learning_rate,
                            weight_decay, metric_for_best_model, seed):
    """
    Set up the training arguments for a Hugging Face Trainer.

    Evaluation and checkpoint saving happen at the end of each
    epoch, the best checkpoint is loaded at the end of training,
    and the model is set up to be pushed to the Hugging Face Hub.

    Args:
        output_dir (str): directory for checkpoints and logs
        epochs (int): number of training epochs
        per_device_train_batch_size (int): training batch size
        per_device_eval_batch_size (int): evaluation batch size
        learning_rate (float): learning rate for the optimizer
        weight_decay (float): weight decay for regularization
        metric_for_best_model (str): metric to pick the best checkpoint
        seed (int): random seed for reproducibility

    Returns:
        transformers.TrainingArguments: the training configuration
    """
    return transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        push_to_hub=True,
        seed=seed
    )
