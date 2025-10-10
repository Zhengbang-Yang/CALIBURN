#!/usr/bin/env python3
"""
referenced https://muse-bench.github.io/
"""

import json
import torch
import argparse
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import os
from rouge_score import rouge_scorer
from scipy.stats import bootstrap
import numpy as np


class RougeEvalLogger:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=False
        )
        self.history = []

    def log(self, prompt: str, gt: str, output: str, question: str | None = None):
        score = self.scorer.score(gt, output)
        d = {
            'prompt': prompt,
            'gt': gt,
            'response': output,
            'rougeL': score['rougeL'].fmeasure,
            'rougeL_recall': score['rougeL'].recall,
            'rouge1': score['rouge1'].fmeasure,
            'rouge1_recall': score['rouge1'].recall
        }
        if question is not None: 
            d['question'] = question
        self.history.append(d)

    def report(self) -> Tuple[Dict, Dict]:
        agg = {}
        for key, val in self.history[0].items():
            if isinstance(val, str): 
                continue
            vals: List[float] = [item[key] for item in self.history]
            agg[f"max_{key}"] = max(vals)
            agg[f"mean_{key}"] = sum(vals) / len(vals)
            agg[f"{key}_ci_lo"], agg[f"{key}_ci_hi"] = bootstrap(
                (vals,), np.mean,
                confidence_level=0.95,
                method='percentile'
            ).confidence_interval
        return agg, self.history


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    """清理输出文本"""
    case_insensitive_words = []
    for word in words:
        case_insensitive_words.append(word.lower())
        case_insensitive_words.append(word.capitalize())
    
    all_words = words + case_insensitive_words
    for word in all_words:
        string = string.split(word)[0]
    return string


def eval_knowmem_fast(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens: int = 32,
    batch_size: int = 8,
    max_icl_examples: int = 3  # 限制ICL示例数量
):
    """优化的KnowMem评估函数，支持批处理"""
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    # Few-shot prompting - 只使用前几个示例
    icl_examples = min(len(icl_qs), max_icl_examples)
    for i in range(icl_examples):
        question, answer = icl_qs[i], icl_as[i]
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    # 批处理生成
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
        batch_questions = questions[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]
        
        # 构建批处理提示
        batch_prompts = []
        for question in batch_questions:
            prompt = general_prompt + f"Question: {question}\nAnswer: "
            batch_prompts.append(prompt)

        # 批处理编码
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        # 记录原始输入长度
        input_lengths = [len(ids) for ids in inputs.input_ids]

        # 批处理生成
        with torch.no_grad():  # 禁用梯度计算
            # 对于多GPU模型，使用第一个GPU作为主要设备
            if hasattr(model, 'hf_device_map') or hasattr(model, 'device_map'):
                # 多GPU模型，使用第一个GPU
                device = torch.device('cuda:0')
            else:
                # 单GPU模型，使用模型所在设备
                device = model.device
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 使用更保守的生成设置
            output_ids = model.generate(
                inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # 启用缓存以加速
                repetition_penalty=1.0  # 避免重复
            )
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 处理每个输出
        for j, (prompt, answer) in enumerate(zip(batch_prompts, batch_answers)):
            # 提取新生成的部分
            new_tokens = output_ids[j, input_lengths[j]:]
            new_output = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # 清理输出
            new_output = get_prefix_before_words_occur(new_output, ["\n\n", "\nQuestion", "Question:"])
            logger.log(prompt, answer, new_output, question=batch_questions[j])

    return logger.report()


def load_model(model_dir: str):
    """加载模型"""
    print(f"Loading model from: {model_dir}")
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # 检查是否是70B模型或17B模型，如果是则使用内存优化
    if "70B" in model_dir or "70b" in model_dir or "17B" in model_dir or "17b" in model_dir:
        print("Detected large model, using memory optimization...")
        
        # 为17B模型设置更保守的内存限制
        if "17B" in model_dir or "17b" in model_dir:
            if num_gpus >= 8:
                # 使用8个GPU，每个分配8GB（为推理预留更多空间）
                max_memory = {i: "8GB" for i in range(8)}
                max_memory["cpu"] = "100GB"
                print(f"Using 8 GPUs for 17B model: {max_memory}")
            elif num_gpus >= 4:
                # 使用4个GPU，每个分配15GB（为推理预留更多空间）
                max_memory = {0: "15GB", 1: "15GB", 2: "15GB", 3: "15GB", "cpu": "100GB"}
                print(f"Using 4 GPUs for 17B model: {max_memory}")
            elif num_gpus >= 2:
                # 使用2个GPU，每个分配30GB（为推理预留更多空间）
                max_memory = {0: "30GB", 1: "30GB", "cpu": "100GB"}
                print(f"Using 2 GPUs for 17B model: {max_memory}")
            else:
                # 单GPU，分配60GB（为推理预留20GB空间）
                max_memory = {0: "60GB", "cpu": "100GB"}
                print(f"Using single GPU for 17B model: {max_memory}")
        else:
            # 70B模型的内存设置
            if num_gpus >= 8:
                # 使用8个GPU，每个分配20GB（70B模型需要更多内存）
                max_memory = {i: "20GB" for i in range(8)}
                max_memory["cpu"] = "100GB"
                print(f"Using 8 GPUs for 70B model: {max_memory}")
            elif num_gpus >= 4:
                # 使用4个GPU，每个分配20GB（更保守的内存分配）
                max_memory = {0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB", "cpu": "100GB"}
                print(f"Using 4 GPUs for 70B model: {max_memory}")
            elif num_gpus >= 2:
                # 使用2个GPU，每个分配40GB（更保守的内存分配）
                max_memory = {0: "40GB", 1: "40GB", "cpu": "100GB"}
                print(f"Using 2 GPUs for 70B model: {max_memory}")
            else:
                # 单GPU无法运行70B模型
                raise ValueError("70B model requires at least 2 GPUs to run")
        
        # 使用更简单的加载方法避免分布式训练问题
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                offload_folder="offload",
                max_memory=max_memory,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load with device_map='auto', trying without: {e}")
            # 如果自动设备映射失败，尝试手动加载
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder="offload",
                max_memory=max_memory,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=None  # 不使用自动设备映射，避免分布式问题
        )
    
    model.eval()
    return model


def load_tokenizer(tokenizer_dir: str):
    """加载tokenizer"""
    print(f"Loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_json(file_path: str):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_csv(data, file_path: str):
    """写入CSV文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(data).to_csv(file_path, index=False)


def eval_model(
    model, tokenizer,
    metrics: List[str],
    corpus: str = 'books',
    knowmem_agg_key: str = 'mean_rougeL',
    knowmem_max_new_tokens: int = 32,
    batch_size: int = 8,
    knowmem_forget_qa_file: str = "distill_unlearn/data/forget_qa.json",
    knowmem_forget_qa_icl_file: str = "distill_unlearn/data/forget_qa_icl.json",
    knowmem_retain_qa_file: str = "distill_unlearn/data/retain_qa.json",
    knowmem_retain_qa_icl_file: str = "distill_unlearn/data/retain_qa_icl.json",
    temp_dir: str | None = None,
) -> Dict[str, float]:
    
    # 确保模型在GPU上（仅对非70B模型且未使用device_map的模型）
    print(f"Model device map: {hasattr(model, 'hf_device_map')}")
    print(f"Model device_map: {hasattr(model, 'device_map')}")
    
    # 对于多GPU模型，避免调用.cuda()以避免内存问题
    if torch.cuda.is_available() and not hasattr(model, 'hf_device_map') and not hasattr(model, 'device_map'):
        print("Moving model to CUDA...")
        try:
            model = model.cuda()
        except Exception as e:
            print(f"Warning: Failed to move model to CUDA: {e}")
            print("Continuing with model on current device...")
    else:
        print("Model already on GPU or using device_map, skipping .cuda() call")
    
    model.eval()  # 设置为评估模式
    
    out = {}
        
    if 'knowmem_f_muse' in metrics:
        print("Evaluating knowmem_f_muse...")
        qa = read_json(knowmem_forget_qa_file)
        # 使用专门的retain_qa_icl.json文件作为ICL示例
        icl = read_json(knowmem_forget_qa_icl_file)
        agg, log = eval_knowmem_fast(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens,
            batch_size=batch_size,
            max_icl_examples=3
        )
        if temp_dir is not None:
            os.makedirs(os.path.join(temp_dir, "knowmem_f_muse"), exist_ok=True)
            with open(os.path.join(temp_dir, "knowmem_f_muse/agg.json"), 'w') as f:
                json.dump(agg, f, indent=2)
            with open(os.path.join(temp_dir, "knowmem_f_muse/log.json"), 'w') as f:
                json.dump(log, f, indent=2)
        out['knowmem_f_muse'] = agg[knowmem_agg_key] * 100
    return out


def load_then_eval_models(
    model_dirs: List[str],
    names: List[str],
    corpus: str = 'books',
    tokenizer_dir: str = "meta-llama/Llama-3.2-3B-Instruct",
    out_file: str | None = None,
    metrics: List[str] = ['knowmem_f_muse'],
    temp_dir: str = "temp",
    batch_size: int = 8
) -> pd.DataFrame:
    
    if not model_dirs:
        raise ValueError(f"`model_dirs` should be non-empty.")
    if len(model_dirs) != len(names):
        raise ValueError(f"`model_dirs` and `names` should equal in length.")
    if out_file is not None and not out_file.endswith('.csv'):
        raise ValueError(f"The file extension of `out_file` should be '.csv'.")

    # Run evaluation
    out = []
    
    for model_dir, name in zip(model_dirs, names):
        print(f"Loading model: {name}")
        model = load_model(model_dir)
        tokenizer = load_tokenizer(tokenizer_dir)
        
        print(f"Evaluating model: {name}")
        res = eval_model(
            model, tokenizer, metrics, corpus,
            temp_dir=os.path.join(temp_dir, name),
            batch_size=batch_size
        )
        out.append({'name': name} | res)
        if out_file is not None: 
            write_csv(out, out_file)
            print(f"Results saved to: {out_file}")
    
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Fast KnowMem Evaluation")
    parser.add_argument('--model_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--names', type=str, nargs='+', default=[])
    parser.add_argument('--tokenizer_dir', type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument('--corpus', type=str, default='books', choices=['books', 'news'])
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--metrics', type=str, nargs='+', default=['knowmem_f_muse'])
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()
    
    args_dict = vars(args)
    load_then_eval_models(**args_dict)


if __name__ == '__main__':
    main() 