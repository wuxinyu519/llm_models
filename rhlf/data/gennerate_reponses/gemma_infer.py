import pickle
import json
from typing import List, Dict, Any
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import os

class GemmaResponseGenerator:
    def __init__(self, gemma_model_path: str = "google/gemma-3-12b-it"):
        """
        初始化Gemma生成器
        Args:
            gemma_model_path: Gemma模型路径或HuggingFace model ID
        """
        print(f"正在加载模型: {gemma_model_path}")
        
       
        self.model_name = gemma_model_path
        
        # vLLM
        self.llm = LLM(
            model=gemma_model_path,
            dtype="bfloat16",
            enforce_eager=True,
            tensor_parallel_size=1,  
            gpu_memory_utilization=0.9,
            max_model_len=2048,  
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        print(f"模型加载完成: {gemma_model_path}")
        
    def load_pkl_files(self, pkl_paths: List[str]) -> List[Dict]:
        """
        加载多个pkl文件
        """
        all_data = []
        for path in pkl_paths:
            print(f"加载文件: {path}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        print(f"总共加载了 {len(all_data)} 条数据")
        return all_data
    
    def generate_tagging_prompt(self, user_instruction: str) -> str:
        """
        生成tagging的prompt
        """
        prompt = f"""You are a tagging system that assigns useful and structured tags to identify the main intentions of user instructions for a helpful AI assistant.

                ANALYSIS PROCESS:
                Step 1 - Domain Analysis
                Step 2 - Task Type Assessment
                Step 3 - Difficulty Evaluation
                Step 4 - Language Detection
                Step 5 - Topic Extraction

                Below is an instruction:  
                [begin] {user_instruction} [end]

                Generate tags following this order:
                Domain → Task Type → Difficulty → Language → Topic(s)

                Please ensure you include safety tags if the content contains:
                - Privacy violations (PII requests, doxxing, etc.)
                - Toxic content (hate speech, harassment, violence, etc.)
                - High-risk requests

                Respond strictly as JSON list:
                [{{"tag": str, "explanation": str}}].
                """
        return prompt

    def generate_with_gemma(self, prompt: str, temperature: float = 0.7, 
                           max_new_tokens: int = 1024) -> str:
        """
        使用Gemma和vLLM生成回答
        """
        # template格式
        messages = [{"role": "user", "content": prompt}]
        
        # 应用chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            max_tokens=max_new_tokens,
        )
        
        # 生成
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        # 提取生成的文本
        response = outputs[0].outputs[0].text.strip()
        
        return response
    
    def generate_multiple_responses(self, prompt: str, n: int = 2) -> List[Dict[str, Any]]:
        """
        生成多个不同的回答
        """
        responses = []
        temperatures = [0.5, 0.9]  # 一个保守，一个创造性
        
        for i, temp in enumerate(temperatures[:n]):
            print(f"  生成回答 {i+1}/{n} (temperature={temp})...")
            try:
                response_text = self.generate_with_gemma(prompt, temperature=temp)
                responses.append({
                    'response_id': i + 1,
                    'temperature': temp,
                    'text': response_text,
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception as e:
                print(f"生成失败: {str(e)}")
                responses.append({
                    'response_id': i + 1,
                    'temperature': temp,
                    'text': '',
                    'error': str(e),
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            time.sleep(0.5)  
        
        return responses
    
    def process_single_sample(self, sample: Dict, index: int, n_responses: int = 2) -> Dict[str, Any]:
       
        print(f"\n{'='*80}")
        print(f"处理样本 {index + 1}")
        print(f"{'='*80}")
        
        # 获取原始prompt
        prompt = sample.get('prompt', '')
        print(f"Prompt: {prompt[:100]}...")
        
        # 生成tagging prompt
        tagging_prompt = self.generate_tagging_prompt(prompt)
        
        # 使用Gemma生成多个回答
        print(f"\n使用Gemma生成 {n_responses} 个回答...")
        responses = self.generate_multiple_responses(tagging_prompt, n=n_responses)
        
        for resp in responses:
            print(f"\n回答{resp['response_id']} (temp={resp['temperature']}):")
            print(f"  {resp['text'][:150]}...")
        
        # 构建结果
        result = {
            'index': sample.get('index', index),
            'prompt': prompt,
            'original_data': sample,
            'tagging_prompt': tagging_prompt,
            'responses': responses,
            'metadata': {
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'gemma_model': self.model_name,
                'n_responses': len(responses)
            }
        }
        
        return result
    
    def process_dataset(self, pkl_paths: List[str], output_path: str, 
                       n_responses: int = 2, batch_size: int = 10, 
                       start_idx: int = 0, end_idx: int = None, limit: int = None):
        """
        处理整个数据集
        Args:
            limit: 限制处理的样本数量（从start_idx开始）
        """
        # 加载数据
        all_data = self.load_pkl_files(pkl_paths)

        if end_idx is None:
            end_idx = len(all_data)
        
        if limit is not None:
            end_idx = min(start_idx + limit, end_idx)
            print(f"处理数量: {limit} 条")
        
        all_data = all_data[start_idx:end_idx]
        print(f"处理范围: [{start_idx}, {end_idx}), 共 {len(all_data)} 条")
        
        processed_data = []
        if os.path.exists(output_path):
            try:
                print(f"检测到已有输出文件，加载中...")
                with open(output_path, 'rb') as f:
                    processed_data = pickle.load(f)
                print(f"已加载 {len(processed_data)} 条已处理数据，将继续追加")
            except Exception as e:
                print(f"加载已有文件失败: {str(e)}，将从头开始")
                processed_data = []
        

        batch_results = [] 
        
        for i, sample in enumerate(tqdm(all_data, desc="生成进度")):
            try:
                result = self.process_single_sample(
                    sample, 
                    start_idx + i, 
                    n_responses=n_responses
                )
                batch_results.append(result)
                
         
                if len(batch_results) >= batch_size:
                    processed_data.extend(batch_results) 
                    self.save_results(processed_data, output_path)
                    print(f"\nBatch保存: 本批 {len(batch_results)} 条，总计 {len(processed_data)} 条")
                    batch_results = []  
                
            except Exception as e:
                print(f"处理样本 {start_idx + i} 出错: {str(e)}")
                batch_results.append({
                    'index': start_idx + i,
                    'error': str(e),
                    'original_data': sample
                })
                continue
        
        if batch_results:
            processed_data.extend(batch_results)
            self.save_results(processed_data, output_path)
            print(f"\n最终保存: 本批 {len(batch_results)} 条，总计 {len(processed_data)} 条")
        
        print(f"\n生成完成！共处理 {len(processed_data)} 条数据")
        print(f"输出文件: {output_path}")
        
        return processed_data
    
    def save_results(self, data: List[Dict], output_path: str):
        """
        保存结果
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        json_path = output_path.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='使用Gemma和vLLM生成tagging回答')
    parser.add_argument('--input', nargs='+', required=True, help='输入pkl文件路径（可以多个）')
    parser.add_argument('--output', required=True, help='输出pkl文件路径')
    parser.add_argument('--model', default='google/gemma-3-12b-it', help='Gemma模型路径')
    parser.add_argument('--n_responses', type=int, default=2, help='每个样本生成几个回答')
    parser.add_argument('--batch_size', type=int, default=10, help='每多少个样本保存一次')
    parser.add_argument('--start_idx', type=int, default=0, help='开始索引')
    parser.add_argument('--end_idx', type=int, default=None, help='结束索引')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的样本数量')
    
    args = parser.parse_args()
    
    generator = GemmaResponseGenerator(gemma_model_path=args.model)
    
    # 处理数据
    generator.process_dataset(
        pkl_paths=args.input,
        output_path=args.output,
        n_responses=args.n_responses,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        limit=args.limit
    )


if __name__ == "__main__":
    main()