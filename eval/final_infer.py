#!/usr/bin/env python3
import os
import pickle
import json
import re
import numpy as np
from tqdm import tqdm
import argparse
import time
import glob
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams
# ============================================================================
# 模型加载和推理部分
# ============================================================================

def is_huggingface_model_id(model_path: str) -> bool:
    return not os.path.exists(model_path)

def get_model_name_from_path(model_path: str) -> str:
    if is_huggingface_model_id(model_path):
        return model_path.replace('/', '_').replace('\\', '_')
    else:
        return os.path.basename(os.path.normpath(model_path))

def truncate_context(context: str, tokenizer, max_tokens: int = 600) -> str:
    """保留前300和后300 tokens"""
    max_each = max_tokens // 2
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return context
    
    front_tokens = tokens[:max_each]
    back_tokens = tokens[-max_each:]
    front_text = tokenizer.decode(front_tokens, skip_special_tokens=True)
    back_text = tokenizer.decode(back_tokens, skip_special_tokens=True)
    
    return front_text + "\n\n[Content truncated]\n\n" + back_text

def load_gemma3_model(model_path, use_quantization=True, device_map="auto"):
    print(f"Loading model with vLLM: {model_path}")
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # vLLM 
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,  
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1, 
    )
    
    print(" vLLM model loaded")
    return model, tokenizer

def extract_tags_with_explanations(tags_text):
    """提取标签和解释"""
    try:
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, tags_text, re.DOTALL)
        
        if json_matches:
            json_str = json_matches[-1]
            try:
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, list):
                    valid_tags = []
                    for item in parsed_json:
                        if isinstance(item, dict) and "tag" in item and "explanation" in item:
                            valid_tags.append({
                                "tag": str(item["tag"]).strip(),
                                "explanation": str(item["explanation"]).strip()
                            })
                    return valid_tags
            except json.JSONDecodeError:
                pass
        
        single_json_pattern = r'\{[^{}]*"tag"[^{}]*"explanation"[^{}]*\}'
        single_matches = re.findall(single_json_pattern, tags_text)
        
        if single_matches:
            valid_tags = []
            for match in single_matches:
                try:
                    item = json.loads(match)
                    if "tag" in item and "explanation" in item:
                        valid_tags.append({
                            "tag": str(item["tag"]).strip(),
                            "explanation": str(item["explanation"]).strip()
                        })
                except:
                    continue
            return valid_tags
        
        return _fallback_parse(tags_text)
        
    except Exception as e:
        print(f"Error extracting tags: {e}")
        return []

def _fallback_parse(response: str):
    try:
        tags = []
        lines = response.split('\n')
        current_tag = None
        current_explanation = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'tag' in line.lower() and ':' in line:
                if current_tag and current_explanation:
                    tags.append({"tag": current_tag, "explanation": current_explanation})
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_tag = parts[1].strip().replace('"', '').replace("'", "")
                    current_explanation = None
            
            elif 'explanation' in line.lower() and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_explanation = parts[1].strip().replace('"', '').replace("'", "")
            
            elif current_tag and not current_explanation and line:
                current_explanation = line.replace('"', '').replace("'", "")
        
        if current_tag and current_explanation:
            tags.append({"tag": current_tag, "explanation": current_explanation})
        
        return tags if tags else [{"tag": "General", "explanation": "Unable to parse"}]
    except:
        return [{"tag": "Error", "explanation": "Failed to parse"}]

def run_inference(model, tokenizer, data, output_file, batch_size=8, save_interval=50):
    """ vLLM """
    print(f"Running vLLM inference on {len(data)} samples...")
    
    start_time = time.time()
    all_results = []
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=512,
        stop=["</s>", "\n\n\n"],
    )
    
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[start:start + batch_size]
        
        # 准备输入
        input_texts = []
        for item in batch:
            inference_text = item.get('inference_context', item.get('input', ''))
            truncated_context = truncate_context(inference_text, tokenizer, max_tokens=300)
            
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text", 
                    # "text": f"You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}. Query: {truncated_context} Assistant:"
                    "text": f"You are a helpful assistant. For the user query below, generate tags in this order: 1) Domain, 2) Task Type, 3) Difficulty, 4) Language, 5) Topics (can be multiple). Explain each tag briefly. Output must be JSON: {{\"tag\": str, \"explanation\": str}}.\n\nQuery: {truncated_context}"
                }]
            }]
            
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                input_texts.append(formatted_prompt)
            except:
                user_content = messages[0]["content"][0]["text"]
                manual_prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
                input_texts.append(manual_prompt)
        
        try:
            outputs = model.generate(input_texts, sampling_params)
            generated_texts = [output.outputs[0].text for output in outputs]
        except Exception as e:
            print(f"Error during generation: {e}")
            generated_texts = ["Generation failed"] * len(batch)
        
        for i, item in enumerate(batch):
            try:
                raw_text = generated_texts[i].strip() if i < len(generated_texts) else "Generation failed"
                parsed_tags = extract_tags_with_explanations(raw_text)
                
                result = {
                    **item,
                    'truncated_input': truncate_context(
                        item.get('inference_context', item.get('input', '')), tokenizer
                    ),
                    'generated_tags': parsed_tags,
                    'raw_response': raw_text,
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                result = {
                    **item,
                    'truncated_input': '',
                    'generated_tags': [],
                    'raw_response': f"Error: {str(e)}",
                }
                all_results.append(result)
    
        if len(all_results) >= save_interval or (start + batch_size) >= len(data):
            with open(output_file, 'wb') as f:
                pickle.dump(all_results, f)
    
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    return all_results




# ============================================================================
# eval: acc
# ============================================================================

class TagEvaluator:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        try:
            self.sentence_model = SentenceTransformer('whaleloops/phrase-bert', device=self.device)
            print(f"Loaded PHRASEBERT on {self.device}")
            
            if self.device == 'cuda':
                dummy_text = ["test"]
                _ = self.sentence_model.encode(dummy_text)
                print("GPU warmed up")
                
        except Exception as e:
            print(f"Warning: Could not load PHRASEBERT: {e}")
            self.sentence_model = None
            
        self.embedding_cache = {}

    def get_embeddings(self, tags):
        if not tags:
            return np.array([])
            
        new_tags = [tag for tag in tags if tag not in self.embedding_cache]
        
        if new_tags:
            try:
                new_embeddings = self.sentence_model.encode(
                    new_tags,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )
                for tag, emb in zip(new_tags, new_embeddings):
                    self.embedding_cache[tag] = emb
            except Exception as e:
                print(f"Error encoding tags: {e}")
                return np.array([])
        
        return np.array([self.embedding_cache[tag] for tag in tags])

    def calculate_exact_match_f1(self, pred_tags, gold_tags):
        """精确匹配 F1"""
        if not pred_tags and not gold_tags:
            return 1.0
        if not pred_tags or not gold_tags:
            return 0.0
        
        pred_set = set([tag.lower().strip() for tag in pred_tags])
        gold_set = set([tag.lower().strip() for tag in gold_tags])
        intersection = pred_set & gold_set

        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(gold_set) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def calculate_semantic_accuracy(self, pred_tags, gold_tags, threshold=0.8):
        """语义准确率（GT->Pred）"""
        if self.sentence_model is None:
            return 0.0, []
        if not pred_tags and not gold_tags:
            return 1.0, []
        if not pred_tags or not gold_tags:
            return 0.0, []
            
        try:
            pred_embeddings = self.get_embeddings(pred_tags)
            gold_embeddings = self.get_embeddings(gold_tags)
            
            if pred_embeddings.size == 0 or gold_embeddings.size == 0:
                return 0.0, []
            
            if self.device == 'cuda' and torch.cuda.is_available():
                pred_tensor = torch.tensor(pred_embeddings, device='cuda')
                gold_tensor = torch.tensor(gold_embeddings, device='cuda')
                sim_matrix = torch.cosine_similarity(
                    gold_tensor.unsqueeze(1), 
                    pred_tensor.unsqueeze(0), 
                    dim=2
                )
                matched = (sim_matrix > threshold).any(dim=1)
                accuracy = matched.sum().float() / len(gold_tags)
                sim_matrix_np = sim_matrix.cpu().numpy()
                accuracy = accuracy.cpu().item()
            else:
                sim_matrix_np = cosine_similarity(gold_embeddings, pred_embeddings)
                matched = (sim_matrix_np > threshold).any(axis=1)
                accuracy = matched.sum() / len(gold_tags)
            
            max_sim_pairs = []
            for gt_idx, gt_tag in enumerate(gold_tags):
                max_pred_idx = np.argmax(sim_matrix_np[gt_idx])
                max_similarity = sim_matrix_np[gt_idx, max_pred_idx]
                max_sim_pairs.append({
                    'gt_tag': gt_tag,
                    'pred_tag': pred_tags[max_pred_idx],
                    'similarity': float(max_similarity)
                })
            
            return accuracy, max_sim_pairs
                
        except Exception as e:
            print(f"Error calculating semantic accuracy: {e}")
            return 0.0, []

    def calculate_semantic_f1(self, pred_tags, gold_tags, threshold=0.8):
        """语义匹配 F1"""
        if self.sentence_model is None or not pred_tags or not gold_tags:
            return 0.0
            
        try:
            pred_embeddings = self.get_embeddings(pred_tags)
            gold_embeddings = self.get_embeddings(gold_tags)
            
            if pred_embeddings.size == 0 or gold_embeddings.size == 0:
                return 0.0
            
            if self.device == 'cuda' and torch.cuda.is_available():
                pred_tensor = torch.tensor(pred_embeddings, device='cuda')
                gold_tensor = torch.tensor(gold_embeddings, device='cuda')
                sim_matrix = torch.cosine_similarity(
                    gold_tensor.unsqueeze(1), 
                    pred_tensor.unsqueeze(0), 
                    dim=2
                )
                sim_matrix_np = sim_matrix.cpu().numpy()
            else:
                sim_matrix_np = cosine_similarity(gold_embeddings, pred_embeddings)
            
            gt_matched = (sim_matrix_np > threshold).any(axis=1).sum()
            recall = gt_matched / len(gold_tags)
            
            pred_matched = (sim_matrix_np > threshold).any(axis=0).sum()
            precision = pred_matched / len(pred_tags)
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1
                
        except Exception as e:
            print(f"Error calculating semantic F1: {e}")
            return 0.0

    def calculate_precision_recall(self, pred_tags, gold_tags, threshold=0.8):
        """计算 Precision 和 Recall"""
        # 精确匹配
        if not pred_tags and not gold_tags:
            em_precision = em_recall = 1.0
        elif not pred_tags or not gold_tags:
            em_precision = em_recall = 0.0
        else:
            pred_set = set([tag.lower().strip() for tag in pred_tags])
            gold_set = set([tag.lower().strip() for tag in gold_tags])
            intersection = pred_set & gold_set
            em_precision = len(intersection) / len(pred_set) if pred_set else 0
            em_recall = len(intersection) / len(gold_set) if gold_set else 0
        
        # 语义匹配
        if self.sentence_model is None or not pred_tags or not gold_tags:
            sem_precision = sem_recall = 0.0
        else:
            try:
                pred_embeddings = self.get_embeddings(pred_tags)
                gold_embeddings = self.get_embeddings(gold_tags)
                
                if pred_embeddings.size == 0 or gold_embeddings.size == 0:
                    sem_precision = sem_recall = 0.0
                else:
                    if self.device == 'cuda' and torch.cuda.is_available():
                        pred_tensor = torch.tensor(pred_embeddings, device='cuda')
                        gold_tensor = torch.tensor(gold_embeddings, device='cuda')
                        sim_matrix = torch.cosine_similarity(
                            gold_tensor.unsqueeze(1), 
                            pred_tensor.unsqueeze(0), 
                            dim=2
                        )
                        sim_matrix_np = sim_matrix.cpu().numpy()
                    else:
                        sim_matrix_np = cosine_similarity(gold_embeddings, pred_embeddings)
                    
                    pred_matched = (sim_matrix_np > threshold).any(axis=0).sum()
                    gold_matched = (sim_matrix_np > threshold).any(axis=1).sum()
                    
                    sem_precision = pred_matched / len(pred_tags) if pred_tags else 0.0
                    sem_recall = gold_matched / len(gold_tags) if gold_tags else 0.0
                    
            except Exception as e:
                print(f"Error calculating precision/recall: {e}")
                sem_precision = sem_recall = 0.0
        
        return em_precision, em_recall, sem_precision, sem_recall

def extract_tags_from_explanations(tags_explanations):
    """从标签解释中提取标签名"""
    if not tags_explanations:
        return []
    tags = []
    for item in tags_explanations:
        if isinstance(item, dict) and 'tag' in item:
            tags.append(item['tag'])
        elif isinstance(item, str):
            tags.append(item)
    return tags

def evaluate_results(results, evaluator):
    """评估单个文件的结果"""
    total_samples = 0
    valid_samples = 0
    
    valid_pred_tags_list = []
    valid_gold_tags_list = []
    failed_cases = []

    for idx, result in enumerate(results):
        if 'error' in result:
            continue
        total_samples += 1

        pred_tags = extract_tags_from_explanations(result.get('generated_tags', []))
        gold_tags = extract_tags_from_explanations(result.get('parsed_tags', []))

        if gold_tags:
            valid_samples += 1
            valid_pred_tags_list.append(pred_tags)
            valid_gold_tags_list.append(gold_tags)
            
            em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
            sem_acc, max_sim_pairs = evaluator.calculate_semantic_accuracy(pred_tags, gold_tags)
            
            if em_f1 < 0.5 or sem_acc < 0.5:
                failed_cases.append({
                    'sample_idx': idx,
                    'predicted_tags': pred_tags,
                    'ground_truth_tags': gold_tags,
                    'em_f1': em_f1,
                    'semantic_accuracy': sem_acc,
                    'max_similarity_pairs': max_sim_pairs,
                    'truncated_input': result.get('truncated_input', 'N/A')[:300] + '...'
                })

    if valid_samples == 0:
        return None


    all_em_f1 = []
    all_sem_acc = []
    all_sem_f1 = []
    all_em_prec = []
    all_em_rec = []
    all_sem_prec = []
    all_sem_rec = []
    
    for pred_tags, gold_tags in zip(valid_pred_tags_list, valid_gold_tags_list):
        em_f1 = evaluator.calculate_exact_match_f1(pred_tags, gold_tags)
        sem_acc, _ = evaluator.calculate_semantic_accuracy(pred_tags, gold_tags)
        sem_f1 = evaluator.calculate_semantic_f1(pred_tags, gold_tags)
        em_prec, em_rec, sem_prec, sem_rec = evaluator.calculate_precision_recall(pred_tags, gold_tags)
        
        all_em_f1.append(em_f1)
        all_sem_acc.append(sem_acc)
        all_sem_f1.append(sem_f1)
        all_em_prec.append(em_prec)
        all_em_rec.append(em_rec)
        all_sem_prec.append(sem_prec)
        all_sem_rec.append(sem_rec)

    metrics = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'exact_match_f1': np.mean(all_em_f1),
        'exact_match_precision': np.mean(all_em_prec),
        'exact_match_recall': np.mean(all_em_rec),
        'semantic_accuracy': np.mean(all_sem_acc),
        'semantic_f1': np.mean(all_sem_f1),
        'semantic_precision': np.mean(all_sem_prec),
        'semantic_recall': np.mean(all_sem_rec),
        'failed_cases_count': len(failed_cases)
    }

    return metrics, failed_cases[:5]  # 返回前5个失败案例

# ============================================================================
# 数据加载
# ============================================================================

def load_single_file(file_path: str, num_samples: int = None):
    """从单个文件加载数据（支持 .json、.jsonl、.pkl）"""
    try:
        file_data = []
        
        
        if file_path.endswith('.pkl'):
            print(f"Loading PKL file: {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    processed_item = process_item(item, file_path, idx)
                    if processed_item:
                        file_data.append(processed_item)
            elif isinstance(data, dict):
                processed_item = process_item(data, file_path, 1)
                if processed_item:
                    file_data.append(processed_item)
            else:
                print(f"Warning: PKL file contains unsupported type: {type(data)}")
                return []
        
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        processed_item = process_item(item, file_path, line_num)
                        if processed_item:
                            file_data.append(processed_item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON decode error at line {line_num}: {e}")
                        continue
        
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for idx, item in enumerate(data, 1):
                        processed_item = process_item(item, file_path, idx)
                        if processed_item:
                            file_data.append(processed_item)
                elif isinstance(data, dict):
                    processed_item = process_item(data, file_path, 1)
                    if processed_item:
                        file_data.append(processed_item)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {file_path}: {e}")
                return []
        
        if num_samples is not None and num_samples > 0:
            file_data = file_data[:num_samples]
        
        print(f"Loaded {len(file_data)} samples from {os.path.basename(file_path)}")
        return file_data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_item(item, file_name, line_num):
    if not isinstance(item, dict):
        print(f"Warning: Item at line {line_num} is not a dict")
        return None
    
    if 'input' in item:
        input_content = item['input']
    elif 'prompt' in item:
        input_content = item['prompt']
    else:
        print(f"Warning: Line {line_num} missing 'input' or 'prompt' field")
        return None
    
    context_content = item.get('context', '')
    # input_content = item['input']
    
    if context_content:
        item['inference_context'] = f"{input_content}\n\n{context_content}"
    else:
        item['inference_context'] = input_content
    
    return item

def find_json_files(data_dir: str):
    """查找所有.json、.jsonl 和 .pkl 文件"""
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        return []
    
    # 查找 JSON 和 JSONL
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json*"), recursive=True)
    
    # 查找 PKL
    pkl_files = glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
    
    all_files = json_files + pkl_files
    
    if not all_files:
        print(f"No .json, .jsonl or .pkl files found in {data_dir}")
        return []
    
    print(f"Found {len(json_files)} JSON files and {len(pkl_files)} PKL files")
    return all_files

# ============================================================================
# main
# ============================================================================

def process_single_file(model, tokenizer, evaluator, json_file, output_dir, args):
    """处理单个JSON文件：推理 + acc"""
    rel_path = os.path.relpath(json_file, args.data_dir)
    rel_dir = os.path.dirname(rel_path)
    file_basename = os.path.splitext(os.path.basename(rel_path))[0]
    
    output_subdir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
    os.makedirs(output_subdir, exist_ok=True)
    
    output_file = os.path.join(output_subdir, f"{file_basename}_results.pkl")
    metrics_file = os.path.join(output_subdir, f"{file_basename}_metrics.json")
    
    print(f"\n{'='*80}")
    print(f"Processing: {rel_path}")
    print(f"{'='*80}")
    
    # 加载数据
    data = load_single_file(json_file, args.num_samples)
    if not data:
        print(f"No data loaded from {json_file}")
        return None
    
    print(f"Loaded {len(data)} samples")
    
    # 运行推理
    try:
        results = run_inference(
            model, tokenizer, data, output_file,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    
    # 评估结果
    print(f"\nEvaluating {file_basename}...")
    eval_result = evaluate_results(results, evaluator)
    
    if eval_result is None:
        print("No valid samples for evaluation")
        return None
    
    metrics, failed_cases = eval_result
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"RESULTS FOR: {file_basename}")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Exact Match F1: {metrics['exact_match_f1']:.3f}")
    print(f"Exact Match Precision: {metrics['exact_match_precision']:.3f}")
    print(f"Exact Match Recall: {metrics['exact_match_recall']:.3f}")
    print(f"Semantic Accuracy: {metrics['semantic_accuracy']:.3f}")
    print(f"Semantic F1: {metrics['semantic_f1']:.3f}")
    print(f"Semantic Precision: {metrics['semantic_precision']:.3f}")
    print(f"Semantic Recall: {metrics['semantic_recall']:.3f}")
    
    # 保存指标
    detailed_metrics = {
        'file_name': file_basename,
        'metrics': metrics,
        'top_failed_cases': failed_cases
    }
    
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_file}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
    
    return {
        'file_name': file_basename,
        'metrics': metrics,
        'output_file': output_file,
        'metrics_file': metrics_file
    }

def main():
    parser = argparse.ArgumentParser(description="Unified inference and evaluation pipeline")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Local model directory or HuggingFace model ID")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .json/.jsonl files")
    parser.add_argument("--output_prefix", type=str, default="results",
                        help="Prefix for output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit samples per file (default: process all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save interval for incremental saves")
    parser.add_argument("--device", type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help="Device for evaluation")

    args = parser.parse_args()
    
    # 验证输入
    model_path = args.checkpoint_path
    is_hf_model = is_huggingface_model_id(model_path)
    
    if not is_hf_model and not os.path.exists(model_path):
        print(f"Error: Local model path does not exist: {model_path}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # 设置输出目录
    model_name = get_model_name_from_path(model_path)
    output_dir = f"{args.output_prefix}_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    
    # 查找文件
    json_files = find_json_files(args.data_dir)
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"\nFound {len(json_files)} files to process")
    
    # 加载模型
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    model, tokenizer = load_gemma3_model(model_path, use_quantization=False)
    
    print(f"\n{'='*80}")
    print("LOADING EVALUATOR")
    print(f"{'='*80}")
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = TagEvaluator(device=device)
    
    # 处理所有文件
    all_file_results = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*80}")
        print(f"FILE {i}/{len(json_files)}")
        print(f"{'='*80}")
        
        result = process_single_file(
            model, tokenizer, evaluator,
            json_file, output_dir, args
        )
        
        if result:
            all_file_results.append(result)
    
    # 汇总所有文件的结果
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(all_file_results)}/{len(json_files)}")
    
    if all_file_results:
        # 计算平均指标
        avg_metrics = {
            'exact_match_f1': np.mean([r['metrics']['exact_match_f1'] for r in all_file_results]),
            'exact_match_precision': np.mean([r['metrics']['exact_match_precision'] for r in all_file_results]),
            'exact_match_recall': np.mean([r['metrics']['exact_match_recall'] for r in all_file_results]),
            'semantic_accuracy': np.mean([r['metrics']['semantic_accuracy'] for r in all_file_results]),
            'semantic_f1': np.mean([r['metrics']['semantic_f1'] for r in all_file_results]),
            'semantic_precision': np.mean([r['metrics']['semantic_precision'] for r in all_file_results]),
            'semantic_recall': np.mean([r['metrics']['semantic_recall'] for r in all_file_results]),
            'total_samples': sum([r['metrics']['total_samples'] for r in all_file_results]),
            'total_valid_samples': sum([r['metrics']['valid_samples'] for r in all_file_results])
        }
        
        print(f"\nAVERAGE METRICS ACROSS ALL FILES:")
        print(f"Total samples: {avg_metrics['total_samples']}")
        print(f"Total valid samples: {avg_metrics['total_valid_samples']}")
        print(f"Exact Match F1: {avg_metrics['exact_match_f1']:.3f}")
        print(f"Exact Match Precision: {avg_metrics['exact_match_precision']:.3f}")
        print(f"Exact Match Recall: {avg_metrics['exact_match_recall']:.3f}")
        print(f"Semantic Accuracy: {avg_metrics['semantic_accuracy']:.3f}")
        print(f"Semantic F1: {avg_metrics['semantic_f1']:.3f}")
        print(f"Semantic Precision: {avg_metrics['semantic_precision']:.3f}")
        print(f"Semantic Recall: {avg_metrics['semantic_recall']:.3f}")
        
        # 按文件显示结果
        print(f"\nPER-FILE RESULTS:")
        print(f"{'-'*80}")
        for result in all_file_results:
            m = result['metrics']
            print(f"{result['file_name']:40s} | EM F1: {m['exact_match_f1']:.3f} | Sem Acc: {m['semantic_accuracy']:.3f} | Sem F1: {m['semantic_f1']:.3f}")
        
        # 保存汇总结果
        summary_file = os.path.join(output_dir, "summary_all_files.json")
        summary_data = {
            'overall_metrics': avg_metrics,
            'per_file_results': [
                {
                    'file_name': r['file_name'],
                    'metrics': r['metrics'],
                    'metrics_file': r['metrics_file']
                }
                for r in all_file_results
            ]
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            print(f"\nSummary saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"All results saved in: {output_dir}")

if __name__ == "__main__":
    main()