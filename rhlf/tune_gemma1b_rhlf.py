#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Using DPO, reward is actually the log-probability difference between chosen and rejected responses.
"""
import os, json
from datetime import datetime
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch
import copy


# ---------- Prompt Template ----------
PROMPT_TEMPLATE = (
    "You are a helpful assistant. For the user query below, generate tags in this order: "
    "1) Domain, 2) Task Type, 3) Difficulty, 4) Language, 5) Topics (can be multiple). "
    "Explain each tag briefly. Output must be JSON: {{\"tag\": str, \"explanation\": str}}.\n\n"
    "Query: {query}"
)


# ---------- Weighted DPO ----------
class WeightedDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        if "weight" in inputs:
            weights = inputs["weight"].to(loss.device).float()
            
            if hasattr(outputs, "losses") and outputs.losses is not None:
                loss = (outputs.losses * weights).mean()
            else:
            
                loss = loss * weights.mean()
        
        return (loss, outputs) if return_outputs else loss



def format_tags_to_json(tags):

    if isinstance(tags, str):
        try:
            json.loads(tags)
            return tags
        except:
            return tags
    
    if isinstance(tags, list):
        formatted_tags = []
        for item in tags:
            if isinstance(item, dict):
                tag_dict = {
                    "tag": item.get("tag", ""),
                    "explanation": item.get("explanation", "")
                }
                formatted_tags.append(tag_dict)
            elif isinstance(item, str):
                formatted_tags.append({
                    "tag": item,
                    "explanation": ""
                })
        
        return json.dumps(formatted_tags, ensure_ascii=False)
    
    return str(tags)


def format_prompt(query: str) -> str:
    return PROMPT_TEMPLATE.format(query=query)


# ---------- loading ----------
def load_local_jsonl_dir(data_dir: str):
    """
    sft data
    format：
    {
        "prompt": ,
        "chosen": [...],  
        "rejected": [...],  
        "weight":   # samlpe weight
    }
    """
    data_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jsonl")
    ]
    
    datasets_list = []
    total = 0
    
    for file in data_files:
        try:
            with open(file, "r", encoding="utf-8") as fin:
                rows = []
                for i, line in enumerate(fin, 1):
                    try:
                        obj = json.loads(line.strip())
                        
                
                        original_query = obj.get("prompt", "")
                        if not isinstance(original_query, str) or not original_query:
                            continue
                        
                   
                        obj["prompt"] = format_prompt(original_query)
                        
                 
                        obj["chosen"] = format_tags_to_json(obj.get("chosen", []))
                        obj["rejected"] = format_tags_to_json(obj.get("rejected", []))
                        
                  
                        if not obj["chosen"] or not obj["rejected"]:
                            continue
                        
                 
                        obj["weight"] = float(obj.get("weight", 1.0))
                        
                        rows.append(obj)
                        
                    except Exception as e:
                        print(f"跳过 {file} 第{i}行: {e}")
                
                if rows:
                    ds = Dataset.from_list(rows)
                    datasets_list.append(ds)
                    total += len(ds)
                    print(f"{os.path.basename(file)}: {len(ds)} 条样本")
                    
        except Exception as e:
            print(f"无法加载 {file}: {e}")
    
    if not datasets_list:
        raise ValueError("未加载任何 JSONL 数据")
    
    merged = concatenate_datasets(datasets_list)
    print(f"数据集样本数 {len(merged)}")
    

    if len(merged) > 0:
        print("\n" + "="*80)
        print("数据样本示例:")
        print("="*80)
        sample = merged[0]
        print(f"Prompt:\n{sample['prompt']}\n")
        print(f"Chosen (前200字符):\n{sample['chosen'][:200]}...\n")
        print(f"Rejected (前200字符):\n{sample['rejected'][:200]}...\n")
        print(f"Weight: {sample.get('weight', 1.0)}")
        print("="*80 + "\n")
    
    return merged


# ---------- main----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="SFT checkpoint")
    parser.add_argument("--data_dir", required=True, help="训练数据目录")
    parser.add_argument("--output_root", default="./experiment/full_wdpo")
    parser.add_argument("--beta", type=float, default=0.1, 
                       help="DPO beta")
    parser.add_argument("--loss_type", default="sigmoid", 
                       choices=["sigmoid", "hinge", "ipo", "kto_pair"])
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="学习率")
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1800)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_root}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Weighted DPO 训练")
    print("=" * 80)
    print(f"SFT Checkpoint: {args.checkpoint_dir}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Beta: {args.beta}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.per_device_batch} x {args.gradient_accumulation}")
    print("=" * 80 + "\n")

    # ---- 加载 tokenizer ----
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer 加载完成\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        use_cache=False,
        local_files_only=True,
    )
    
    print("reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        use_cache=False,
        local_files_only=True,
    )

    for param in ref_model.parameters():
        param.requires_grad = False
    print("Reference 模型创建完成\n")


    dataset = load_local_jsonl_dir(args.data_dir)

    # ---- DPO ----
    dpo_config = DPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=10,
        save_steps=500,
        save_strategy="steps",
        save_total_limit=3,  
        report_to="tensorboard",
        run_name=f"full_wdpo_{timestamp}",
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        dataset_num_proc=4,
    )

    trainer = WeightedDPOTrainer(
        model=model,
        ref_model=ref_model, 
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 80)
    print("开始 Weighted DPO 训练...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    

    final_output = os.path.join(output_dir, "final")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "checkpoint_dir": args.checkpoint_dir,
            "data_dir": args.data_dir,
            "prompt_template": PROMPT_TEMPLATE,
            "beta": args.beta,
            "loss_type": args.loss_type,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "per_device_batch": args.per_device_batch,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "timestamp": timestamp,
        }, f, indent=2)
    
    print(f"最终模型已保存到 {final_output}")
    print(f"训练配置已保存到 {config_path}")
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()