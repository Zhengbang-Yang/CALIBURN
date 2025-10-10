from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import argparse
import wandb
import json
from torch.utils.data import Dataset



class CATNIPTrainer(Trainer):
    def __init__(self, ref_model: AutoModelForCausalLM | None = None, *args, temperature=1.0, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.beta=beta
        if ref_model:
            device = next(self.model.parameters()).device
            self.ref_model = ref_model
            self.ref_model.to(device)
            print("Using reference model for TWISE")
        else:
            self.ref_model = None
        print("beta:", self.beta)

    def log(self, logs, start_time=None, **kwargs):
        clean = {}
        for k, v in (logs or {}).items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
                v = (v.mean() if v.numel() > 1 else v).float().cpu().item()
            elif hasattr(v, "item") and callable(getattr(v, "item")):
                # numpy scalar 等
                v = v.item()
            clean[k] = v
        return super().log(clean, start_time=start_time, **kwargs)

    def get_batch_loss(self, logits, labels):
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()
        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # get the sum loss for each sequence in a batch
        loss = loss_function(logits.transpose(-1,-2), shifted_labels).sum(dim=-1)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch: int | None = None):
        loss = 0
        device = next(model.parameters()).device
        # if self.ref_model:
        #     self.ref_model.to(device)
        labels = inputs["labels"].to(device)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items() if k != "labels"
        }
        labels = labels[..., 1:].contiguous()
        
        outputs_f = model(**inputs)
        logits_f = outputs_f.logits[..., :-1, :].contiguous()
        mask = (labels != -100)
        labels = labels.clamp_min(0)
        # TWISE
        log_p  = F.log_softmax(logits_f,  dim=-1).clamp_max(0.0)      # log π(y|x)
        log_p = log_p.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        log_p = log_p.masked_fill(~mask, 0.0)
        print(f"log_p shape: {log_p.shape}")
        print(f"mask shape: {mask.shape}")
        if self.ref_model:
            with torch.no_grad():
                outputs_ref = self.ref_model(**inputs)
                logits_ref = outputs_ref.logits[..., :-1, :].contiguous()
            log_p_ref = F.log_softmax(logits_ref,  dim=-1)      # log π^(y|x)
            log_p_ref = log_p_ref.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
            log_p_ref = log_p_ref.masked_fill(~mask, 0.0)
            neg_log_prob_ratio = log_p_ref-log_p
        else:
            with torch.no_grad():
                log_p_hat  = F.log_softmax(outputs_f.logits,  dim=-1).clamp_max(0.0)      # log π^(y|x) 
                log_p_hat = log_p_hat.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1) 
            cutoff = -torch.log(torch.tensor(2.0, device=log_p_hat.device, dtype=log_p_hat.dtype))  # -log(2)
            use_log1p = log_p_hat <= cutoff
            out = torch.empty_like(log_p_hat)
            out[use_log1p]  = torch.log1p(-torch.exp(log_p_hat[use_log1p]))   # x <= -log2
            out[~use_log1p] = torch.log(-torch.expm1(log_p_hat[~use_log1p]))  # -log2 < x <= 0
            log_1m_p_hat = out
            # log_1m_p_hat = torch.log1p(-torch.exp(-neg_log_p_hat).clamp(min=1e-8, max=1 - 1e-8))  
            log_1m_p_hat = log_1m_p_hat.masked_fill(~mask, 0.0)
            neg_log_prob_ratio = log_1m_p_hat - log_p
        
        print(f"neg_log_prob_ratio shape: {neg_log_prob_ratio.shape}")
        print(f"mask sum: {mask.sum(dim=-1)}")
        # TWISE_token
        loss_term=-F.logsigmoid(self.beta * neg_log_prob_ratio)
        loss_term=loss_term.masked_fill(~mask, 0.0)
        loss_term=loss_term.sum(dim=(-1,-2))/mask.sum(dim=(-1,-2))
        loss += loss_term
        print(f"Step {self.state.global_step}: pi(y|x) = {torch.exp(log_p).masked_fill(~mask, 0.0).sum(dim=(-1,-2))/mask.sum(dim=(-1,-2)).item()}")
        self.log({"pi(y|x)": torch.exp(log_p).masked_fill(~mask, 0.0).sum(dim=(-1,-2))/mask.sum(dim=(-1,-2)).detach().cpu().item()})
        if self.ref_model:
            self.log({"pi_ref(y|x)": torch.exp(log_p_ref).masked_fill(~mask, 0.0).sum(dim=(-1,-2))/mask.sum(dim=(-1,-2)).detach().cpu().item()})
        else:
            self.log({"1-pi^(y|x)": 1-torch.exp(log_p_hat).masked_fill(~mask, 0.0).sum(dim=(-1,-2))/mask.sum(dim=(-1,-2)).detach().cpu().item()})
        # # TWISE
        # log_p = log_p.masked_fill(~mask, 0.0)
        # log_p_hat = log_p_hat.masked_fill(~mask, 0.0)
        # loss += -F.logsigmoid(self.beta * neg_log_prob_ratio.sum(dim=-1)/mask.sum(dim=-1)).mean()
        # print(f"Step {self.state.global_step}: pi(y|x) = {(torch.exp(log_p).sum(dim=-1)/mask.sum(dim=-1)).mean().item()}")
        # self.log({"pi(y|x)": (torch.exp(log_p).sum(dim=-1)/mask.sum(dim=-1)).mean().item()})
        # self.log({"1-pi^(y|x)": (1-torch.exp(log_p_hat).sum(dim=-1)/mask.sum(dim=-1)).mean().item()})

        return (loss, outputs_f) if return_outputs else loss




class QADataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        if "jsonl" in path:
            raw_data=[]
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append(json.loads(line))
        else:
            with open(path, "r") as f:
                raw_data = json.load(f)

        self.data = []
        for item in raw_data:
            question = item["question"].strip()
            answer = item["answer"].strip()

            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>Question: {question}\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            full_text = prompt + answer + "<|eot_id|>"

            enc = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            prompt_len = len(tokenizer(prompt, truncation=True)["input_ids"])
            labels = enc["input_ids"].clone()
            labels[0, :prompt_len] = -100
            labels[enc["attention_mask"] == 0] = -100
            # print(labels[0])
            self.data.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": labels[0],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--forget_file", type=str, required=True)
    parser.add_argument("--retain_file", type=str, default=None)  
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)  
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--ref", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name or model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if args.ref:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        # device = next(model.parameters()).device
        ref_model = ref_model.eval()
        # ref_model.to(device)
    else:
        ref_model = None
    train_dataset = QADataset(args.forget_file, tokenizer, max_length=args.max_length)

    # wandb init
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type = "constant",
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_run_name,
        remove_unused_columns=False
    )

    trainer = CATNIPTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        temperature=args.temperature,
        beta=args.beta,
        ref_model=ref_model
    )

    trainer.train()
    trainer.save_model(args.output_dir) 
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()