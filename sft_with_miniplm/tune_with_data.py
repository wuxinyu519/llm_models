import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
import re
from tqdm import tqdm
import os
from datetime import datetime
import argparse
import time
import random
import warnings

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# 检测GPU能力
def get_torch_dtype():
    """根据GPU能力选择最佳数据类型"""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        if capability >= 8:  # A100, H100, RTX 30x+
            return torch.bfloat16, True, False
        else:  # T4, V100, RTX 20x
            return torch.float16, False, True
    return torch.float32, False, False

class PIITaggingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        

        self.template = (
            "You are a helpful assistant. For the user query below, generate tags in this order: "
            "1) Domain, 2) Task Type, 3) Difficulty, 4) Language, 5) Topics (can be multiple). "
            "Explain each tag briefly. Output must be JSON: {{\"tag\": str, \"explanation\": str}}. "
            "Query: {query}\nAssistant: {response}"
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['prompt']
        

        if isinstance(item['parsed_tags'], str):
            try:
                tags = eval(item['parsed_tags'])
            except:
                tags = item['parsed_tags']
        else:
            tags = item['parsed_tags']
        
        if not isinstance(tags, list):
            tags = [tags]
        
        response = json.dumps(tags, ensure_ascii=False)
        

        if self.tokenizer.eos_token:
            response = response + self.tokenizer.eos_token
        

        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        if len(query_tokens) > 1800:
            query_tokens = query_tokens[:1800]
            query = self.tokenizer.decode(query_tokens, skip_special_tokens=True)
        
        text = self.template.format(query=query, response=response)
        

        prompt_text = self.template.format(query=query, response="").rstrip()
        
        prompt_encoding = self.tokenizer(prompt_text, add_special_tokens=False)
        full_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors="pt"
        )
        
        labels = full_encoding['input_ids'].clone()
        
  
        prompt_length = len(prompt_encoding['input_ids'])
        labels[0, :prompt_length] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100
        

        valid_labels = (labels != -100).sum().item()
        if valid_labels == 0:
            print(f"ERROR: Sample {idx} has no valid labels!")

            labels[0, -50:] = full_encoding['input_ids'][0, -50:].clone()
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }

class TaggingEvaluator:
    def __init__(self, device='cuda'):
        try:
            self.phrase_model = SentenceTransformer('whaleloops/phrase-bert')
            self.phrase_model = self.phrase_model.to(device)
        except Exception as e:
            print(f"Warning: Could not load phrase-bert model: {e}")
            print("Using fallback evaluation method")
            self.phrase_model = None
        
    def extract_tags_from_response(self, response: str) -> List[Dict[str, str]]:
        response = response.strip()
        result = []
        
        try:
            if response.startswith('['):
                try:
                    result = json.loads(response)
                    return result
                except json.JSONDecodeError:
     
                    bracket_count = 0
                    end_pos = -1
                    for i, char in enumerate(response):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
                    if end_pos > 0:
                        first_json = response[:end_pos]
                        result = json.loads(first_json)
                        return result
            

            pattern = r'\{"tag":\s*"([^"]+)",\s*"explanation":\s*"([^"]*?)"\}'
            matches = re.findall(pattern, response, re.DOTALL)
            
            if matches:
                result = [{"tag": tag, "explanation": exp} for tag, exp in matches]
                return result
            

            json_objects = []
            current_pos = 0
            
            while current_pos < len(response):
                start_pos = response.find('{"tag":', current_pos)
                if start_pos == -1:
                    break
                
                brace_count = 0
                end_pos = -1
                for i in range(start_pos, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > start_pos:
                    try:
                        json_str = response[start_pos:end_pos]
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    
                    current_pos = end_pos
                else:
                    break
            
            if json_objects:
                return json_objects
                
        except Exception as e:
            print(f"Error parsing response: {e}")
            
        return []
    
    def calculate_exact_match_f1(self, pred_tags: List[str], gold_tags: List[str]) -> float:
        pred_set = set(pred_tags)
        gold_set = set(gold_tags)
        
        if len(pred_set) == 0 and len(gold_set) == 0:
            return 1.0
        if len(pred_set) == 0 or len(gold_set) == 0:
            return 0.0
        
        intersection = pred_set & gold_set
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def calculate_semantic_f1(self, pred_tags: List[str], gold_tags: List[str], threshold=0.8) -> float:
        if self.phrase_model is None:
            return self.calculate_exact_match_f1(pred_tags, gold_tags)
            
        if len(pred_tags) == 0 and len(gold_tags) == 0:
            return 1.0
        if len(pred_tags) == 0 or len(gold_tags) == 0:
            return 0.0
        
        try:
            pred_embeddings = self.phrase_model.encode(pred_tags)
            gold_embeddings = self.phrase_model.encode(gold_tags)
            
            
            similarity_matrix = cosine_similarity(pred_embeddings, gold_embeddings)
            
            # 每个预测标签找最佳匹配
            matched_pred = 0
            for i in range(len(pred_tags)):
                if np.max(similarity_matrix[i]) >= threshold:
                    matched_pred += 1
            
            # 每个真实标签找最佳匹配        
            matched_gold = 0
            for j in range(len(gold_tags)):
                if np.max(similarity_matrix[:, j]) >= threshold:
                    matched_gold += 1
            
            precision = matched_pred / len(pred_tags)
            recall = matched_gold / len(gold_tags)
            
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        except Exception as e:
            print(f"Error in semantic F1 calculation: {e}")
            return self.calculate_exact_match_f1(pred_tags, gold_tags)

def load_data(pkl_path: str, limit_data: int = None):
    processed_data = []
    
    try:

        with open(pkl_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('{'):

                f.seek(0) 
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if item:  
                            processed_data.append(item)
                    except json.JSONDecodeError:
                        continue
            else:
                raise ValueError("JSONL格式加载失败")
                
    except (UnicodeDecodeError, ValueError):
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        for item in data:
            if isinstance(item, dict) and 'prompt' in item and 'parsed_tags' in item:
                processed_data.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                processed_data.append({
                    'index': item[0],
                    'prompt': item[1],
                    'parsed_tags': item[2]
                })
    
    print(f"成功加载 {len(processed_data)} 条数据")
    
    if limit_data is not None and limit_data > 0:
        processed_data = processed_data[:limit_data]
    
    return processed_data
def evaluate_model(model, tokenizer, eval_dataset, evaluator, device, show_samples=True):
    model.eval()
    exact_match_f1_scores = []
    semantic_f1_scores = []
    

    if isinstance(model, nn.DataParallel):
        generate_model = model.module
    else:
        generate_model = model
    
    eval_count = min(3, len(eval_dataset))
    
    with torch.no_grad():
        for i in tqdm(range(eval_count), desc="Evaluating"):
            try:
                query = eval_dataset.data[i]['prompt']
                query_prompt = eval_dataset.template.format(query=query, response="").rstrip()
                
                inputs = tokenizer(query_prompt, return_tensors="pt").to(device)
                

                outputs = generate_model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    use_cache=False,  
                    temperature=0.6, 
                    top_p=0.9, 
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
       
                pred_tags_dicts = evaluator.extract_tags_from_response(response)
                pred_tags = [tag_dict.get('tag', '') for tag_dict in pred_tags_dicts 
                           if isinstance(tag_dict, dict) and 'tag' in tag_dict]
                
                # 获取ground truth tags
                gold_tags_dicts = eval_dataset.data[i]['parsed_tags']
                if isinstance(gold_tags_dicts, str):
                    try:
                        gold_tags_dicts = eval(gold_tags_dicts)
                    except:
                        gold_tags_dicts = []
                
                if not isinstance(gold_tags_dicts, list):
                    gold_tags_dicts = [gold_tags_dicts]
                
                gold_tags = [tag_dict.get('tag', '') for tag_dict in gold_tags_dicts 
                           if isinstance(tag_dict, dict) and 'tag' in tag_dict]
                
    
                if show_samples and i < 3:
                    print(f"\n{'='*80}")
                    print(f"SAMPLE {i+1}:")
                    print(f"{'='*80}")
                    print(f"QUERY: {query_prompt}...")
                    print(f"\nGROUND TRUTH TAGS: {gold_tags}")
                    print(f"\nMODEL RESPONSE:")
                    if pred_tags_dicts:
                        print(json.dumps(pred_tags_dicts, indent=2, ensure_ascii=False))
                    else:
                        print("解析失败，原始输出:")
                        print(response)
                    print(f"\nPREDICTED TAGS: {pred_tags}")
                    
    
                    em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
                    sem_f1 = evaluator.calculate_semantic_f1(pred_tags, gold_tags)
                    print(f"\nSAMPLE METRICS:")
                    print(f"  - Exact Match F1: {em_f1:.3f}")
                    print(f"  - Semantic F1: {sem_f1:.3f}")
                    print(f"{'='*80}")
                

                em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
                sem_f1 = evaluator.calculate_semantic_f1(pred_tags, gold_tags)
                
                exact_match_f1_scores.append(em_f1)
                semantic_f1_scores.append(sem_f1)
                
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
    
    return {
        'exact_match_f1': np.mean(exact_match_f1_scores) if exact_match_f1_scores else 0.0,
        'semantic_f1': np.mean(semantic_f1_scores) if semantic_f1_scores else 0.0
    }

def setup_model_and_tokenizer(model_name, use_quantization=False):
    

    torch_dtype, use_bf16, use_fp16 = get_torch_dtype()
    
    print(f"Using torch_dtype: {torch_dtype}")
    print(f"BF16: {use_bf16}, FP16: {use_fp16}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if hasattr(tokenizer, 'default_chat_template') and tokenizer.chat_template is None:
        tokenizer.chat_template = tokenizer.default_chat_template
    
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False  
    
    print("Disabled use_cache for stable training")
    
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Enabled gradient checkpointing")
    

    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, 'cache_implementation'):
            model.generation_config.cache_implementation = None

    

    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized token embeddings to {len(tokenizer)}")
    
    return model, tokenizer, torch_dtype, use_bf16, use_fp16

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma 3 models for tagging')
    parser.add_argument('--limit_data', type=int, default=None, 
                       help='Limit data samples for testing')
    parser.add_argument('--pkl_path', type=str, default='pii_result.pkl',
                       help='Path to pickle file containing data')
    parser.add_argument('--models', nargs='+', 
                       default=["google/gemma-3-1b-it"],  
                       help='List of model names to fine-tune')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per device batch size')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--use_quantization', action='store_true',
                       help='Use 4-bit quantization')
    parser.add_argument('--use_all_data', action='store_true',
                       help='Use all data for training')
    parser.add_argument('--eval_split', type=float, default=0.2,
                       help='Fraction of data for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability()[0]
        print(f"GPU: {gpu_name} (Compute Capability: {capability}.x)")
    
    print("Loading data...")
    data = load_data(args.pkl_path, args.limit_data)
    print(f"Loaded {len(data)} samples")

    if args.use_all_data:
        print("Using ALL data for training")
        train_data = data
        eval_sample_size = min(50, len(data))
        eval_data = random.sample(data, eval_sample_size)
    else:
        split_idx = int((1 - args.eval_split) * len(data))
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")
    
    evaluator = TaggingEvaluator(device)
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Fine-tuning {model_name}")
        print(f"{'='*60}")
        
        clean_name = model_name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"./experiments/{clean_name}_{timestamp}"
        if args.limit_data:
            model_dir += f"_limit{args.limit_data}"
        
        os.makedirs(f"{model_dir}/results", exist_ok=True)
        print(f"Model directory: {model_dir}")
        
        try:

            model, tokenizer, torch_dtype, use_bf16, use_fp16 = setup_model_and_tokenizer(
                model_name, args.use_quantization
            )
            
      
            train_dataset = PIITaggingDataset(train_data, tokenizer)
            eval_dataset = PIITaggingDataset(eval_data, tokenizer)
            
    
            per_device_batch_size = args.batch_size
            gradient_accumulation_steps = 4
            
 
            total_steps = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps * num_gpus)
            if total_steps == 0:
                gradient_accumulation_steps = max(1, len(train_dataset) // (per_device_batch_size * num_gpus))
                total_steps = max(1, len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps * num_gpus))

            save_steps = max(50, total_steps // 5)
            logging_steps = max(1, total_steps // 10)
            warmup_steps = max(10, total_steps // 20)
            
            print(f"Training configuration:")
            print(f"  - Per device batch size: {per_device_batch_size}")
            print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
            print(f"  - Effective batch size: {per_device_batch_size * gradient_accumulation_steps * num_gpus}")
            print(f"  - Total steps: {total_steps}")
            print(f"  - Warmup steps: {warmup_steps}")
            print(f"  - Data type: {torch_dtype}")
            

            training_args = TrainingArguments(
                output_dir=f"{model_dir}/checkpoints",
                num_train_epochs=args.epochs,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=5e-5,  
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                max_grad_norm=1.0, 
                
                fp16=use_fp16,
                bf16=use_bf16,
                dataloader_pin_memory=False,  
                dataloader_num_workers=0,    

                save_safetensors=False,
                logging_dir=f"{model_dir}/tensorboard_logs",
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_strategy="steps",
                eval_strategy="no",
                load_best_model_at_end=False,
                save_total_limit=3,
                

                remove_unused_columns=False,
                gradient_checkpointing=True,
                

                report_to="tensorboard",
                disable_tqdm=False,
                log_level="info",
                logging_first_step=True,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
                processing_class=tokenizer,
            )
            

            print("\n" + "="*50)
            print("EVALUATING BEFORE TRAINING")
            print("="*50)
            before_metrics = evaluate_model(model, tokenizer, eval_dataset, evaluator, device, show_samples=True)
            print(f"Before Training - EM F1: {before_metrics['exact_match_f1']:.3f}, Semantic F1: {before_metrics['semantic_f1']:.3f}")

            print(f"\n{'='*60}")
            print(f"STARTING TRAINING")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            
            training_time = end_time - start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {hours}h {minutes}m {seconds}s")
            print(f"Average time per step: {training_time/total_steps:.2f}s")
            print(f"{'='*60}")


            print("Saving final model...")
            final_save_path = os.path.join(model_dir, "final_model")
            trainer.save_model(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            print(f"✓ Model saved to: {final_save_path}")

            # 训练后评估
            print("\n" + "="*50)
            print("EVALUATING AFTER TRAINING")
            print("="*50)
            after_metrics = evaluate_model(model, tokenizer, eval_dataset, evaluator, device, show_samples=True)
            print(f"After Training - EM F1: {after_metrics['exact_match_f1']:.3f}, Semantic F1: {after_metrics['semantic_f1']:.3f}")
            
            # 计算改进
            em_improvement = after_metrics['exact_match_f1'] - before_metrics['exact_match_f1']
            sem_improvement = after_metrics['semantic_f1'] - before_metrics['semantic_f1']
            
            print(f"\nIMPROVEMENT SUMMARY:")
            print(f"  - Exact Match F1: {em_improvement:+.3f} ({before_metrics['exact_match_f1']:.3f} → {after_metrics['exact_match_f1']:.3f})")
            print(f"  - Semantic F1: {sem_improvement:+.3f} ({before_metrics['semantic_f1']:.3f} → {after_metrics['semantic_f1']:.3f})")
            
            # 保存结果
            results = {
                "model": model_name,
                "timestamp": timestamp,
                "data_info": {
                    "total_samples": len(data),
                    "train_samples": len(train_data),
                    "eval_samples": len(eval_data),
                    "limit_data": args.limit_data
                },
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": per_device_batch_size,
                    "effective_batch_size": per_device_batch_size * gradient_accumulation_steps * num_gpus,
                    "learning_rate": training_args.learning_rate,
                    "total_steps": total_steps,
                    "torch_dtype": str(torch_dtype),
                    "use_quantization": args.use_quantization,
                    "num_gpus": num_gpus
                },
                "results": {
                    "before_training": before_metrics,
                    "after_training": after_metrics,
                    "improvement": {
                        "exact_match_f1": em_improvement,
                        "semantic_f1": sem_improvement,
                    },
                    "training_time_seconds": training_time,
                    "time_per_step": training_time / total_steps if total_steps > 0 else 0
                }
            }
            
            with open(f"{model_dir}/results/results.json", 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {model_dir}/results/results.json")

            
        except Exception as e:
            print(f"Error during training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {model_name}")
        print(f"{'='*60}")

    print(f"\nAll models completed!")
    print(f"Results saved in ./experiments/ directory")

if __name__ == "__main__":
    main()