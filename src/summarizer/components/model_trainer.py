import torch 
import os
import bitsandbytes as bnb 
import evaluate 
import numpy as np

from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer 
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer 
from datasets import load_dataset,load_from_disk

from src.summarizer.entity.config_entity import ModelTrainerConfig
 
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config


    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        # Handle tuple output
        if isinstance(preds, tuple):
            preds = preds[0]

        # Convert to numpy safely
        preds = np.array(preds)
        labels = np.array(labels)

        # Replace invalid values
        preds[preds < 0] = self.tokenizer.pad_token_id
        preds[preds >= self.tokenizer.vocab_size] = self.tokenizer.pad_token_id

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # Force safe dtype
        preds = preds.astype(np.int64)
        labels = labels.astype(np.int64)

        decoded_preds = self.tokenizer.batch_decode(
            preds.tolist(),   #critical: convert to Python lists
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        decoded_labels = self.tokenizer.batch_decode(
            labels.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        decoded_preds = ["\n".join(p.split(". ")) for p in decoded_preds]
        decoded_labels = ["\n".join(l.split(". ")) for l in decoded_labels]

        rouge = evaluate.load("rouge")  # Load rouge here
        
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        result = {k: round(v * 100, 4) for k, v in result.items()}

        gen_lens = [np.count_nonzero(p != self.tokenizer.pad_token_id) for p in preds]
        result["gen_len"] = float(np.mean(gen_lens))

        return result

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
        self.tokenizer = tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model=model)

        dataset_dict = load_from_disk(self.config.data_path)

        train_frac = self.config.train_fraction
        eval_frac = self.config.eval_fraction

        train_dataset = (
            dataset_dict["train"]
            .shuffle(seed=42)
            .select(range(int(len(dataset_dict["train"]) * train_frac)))
        )

        eval_dataset = (
            dataset_dict["validation"]
            .shuffle(seed=42)
            .select(range(int(len(dataset_dict["validation"]) * eval_frac)))
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir,

            # Core training
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,

            # Optimizer & scheduler
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,

            # Precision & performance
            bf16=self.config.use_bf16,
            tf32=self.config.tf32,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,

            # Logging / eval / saving
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,

            # Misc
            report_to=self.config.report_to,  # avoids wandb unless you want it
            load_best_model_at_end=self.config.load_best_model_at_end,
            generation_num_beams=self.config.generation_num_beams,
            generation_max_length=self.config.generation_max_length,
            predict_with_generate=self.config.predict_with_generate,
            max_grad_norm=self.config.max_grad_norm,
            torch_compile=self.config.torch_compile
        )

        trainer = Seq2SeqTrainer(model=model,
                          args=training_args,
                          processing_class=self.tokenizer,
                          data_collator=data_collator,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          compute_metrics=self.compute_metrics)
        
        trainer.train()

        model.save_pretrained(os.path.join(self.config.root_dir, "model"))
        self.tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))