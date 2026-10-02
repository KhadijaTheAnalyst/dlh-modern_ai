#!/usr/bin/env python3
"""Train, evaluate and save a Hugging Face model."""
import transformers


def train_model(model, training_args, train_dataset, eval_dataset,
                test_dataset, tokenizer, data_collator, compute_metrics,
                model_save_name):
    """
    Fine-tune a model, evaluate it and save it with its tokenizer.

    Args:
        model (transformers.PreTrainedModel): model to fine-tune
        training_args (transformers.TrainingArguments): training setup
        train_dataset (datasets.Dataset): tokenized training data
        eval_dataset (datasets.Dataset): tokenized validation data
        test_dataset (datasets.Dataset): tokenized test data
        tokenizer: tokenizer used for preprocessing
        data_collator (callable): data collator for dynamic padding
        compute_metrics (callable): computes metrics from predictions
        model_save_name (str): directory to save the model and tokenizer

    Returns:
        tuple: (trainer, train_results, test_results)
    """
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_results = trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    trainer.save_model(model_save_name)
    tokenizer.save_pretrained(model_save_name)

    return trainer, train_results, test_results
