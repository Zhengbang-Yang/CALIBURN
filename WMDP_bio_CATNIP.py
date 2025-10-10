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

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch: int | None = None):
        loss = 0
        device = next(model.parameters()).device

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
            log_1m_p_hat = log_1m_p_hat.masked_fill(~mask, 0.0)
            neg_log_prob_ratio = log_1m_p_hat - log_p
        
        print(f"neg_log_prob_ratio shape: {neg_log_prob_ratio.shape}")
        print(f"mask sum: {mask.sum(dim=-1)}")

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

        return (loss, outputs_f) if return_outputs else loss


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            count=0
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                text = obj.get("text")
                if text: self.data.append(text)
                count+=1    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return {"text": self.data[idx]}
        
def make_collate_fn(tokenizer, max_length):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        texts = [ex["abstract"] for ex in batch]

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        keep_mask = torch.zeros_like(labels, dtype=torch.bool)
        keep_mask[:, 16::16] = True
        labels[~keep_mask] = -100
        
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate_fn
    
def _get_decoder_layers(model):
    maybe = getattr(getattr(model, "model", None), "layers", None)
    if isinstance(maybe, (list, tuple)):
        return list(maybe), "model.layers"

    maybe = getattr(getattr(model, "transformer", None), "h", None)
    if isinstance(maybe, (list, tuple)):
        return list(maybe), "transformer.h"

    raise RuntimeError("Can not recognize")


def freeze_only_layers(model, target_ids,param_ids):
    for p in model.parameters():
        p.requires_grad = False

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise RuntimeError("Can not recognize model.model.layers or transformer.h")

    n = len(layers)
    true_ids = set()
    for i in target_ids:
        if i < 0:
            i = n + i
        if not (0 <= i < n):
            raise IndexError(f"layer index {i} out of boundry ({n} layers total)")
        true_ids.add(i)

    for i in true_ids:
        for j, p in enumerate(layers[i].parameters()):
            if param_ids is None or j in param_ids:
                p.requires_grad = True
                print(f"layer {i}, param {j}, shape={tuple(p.shape)}, trainable={p.requires_grad}")

    print(f"unfreezed layers: {sorted(true_ids)}")

    return sorted(true_ids)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--forget_file", type=str, required=True)
    parser.add_argument("--retain_file", type=str, default=None)  
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)  
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="layers to train")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--ref", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    layer_ids = [int(x.strip()) for x in args.layer_ids.split(",") if x.strip()]
    
    
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name or model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        layer_cls_name = model.model.layers[0].__class__.__name__
    except AttributeError:
        layer_cls_name = model.model.decoder.layers[0].__class__.__name__
    print("Auto FSDP wrap class:", layer_cls_name)
    freeze_only_layers(model, target_ids=[5, 6, 7], param_ids=[6])
    
    if args.ref:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        ref_model = ref_model.eval()

    else:
        ref_model = None
    train_dataset = TextDataset(args.forget_file, tokenizer, max_length=args.max_length)
    data_collator = make_collate_fn(tokenizer, args.max_length)
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
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap=layer_cls_name,
    )

    trainer = CATNIPTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator, 
        temperature=args.temperature,
        beta=args.beta,
        ref_model=ref_model
    )

    trainer.train()
    trainer.save_model(args.output_dir) 
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()