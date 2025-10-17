"""
RLHF Data Annotation (No Format Issue Filtering)

Usage:
    python rlhf_annotator.py --input data.jsonl --output annotated.jsonl
"""

import json
import os
import argparse
from openai import OpenAI
from typing import List, Dict, Optional
from tqdm import tqdm
import time


class RLHFDataAnnotator:
    """Simplified annotator without format issue filtering"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL or JSON file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            f.seek(0)
            try:
                json.loads(first_line)
                is_jsonl = True
            except:
                is_jsonl = False
            
            if is_jsonl:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    data = json_data
                elif isinstance(json_data, dict):
                    for key in ['data', 'results', 'items', 'samples']:
                        if key in json_data and isinstance(json_data[key], list):
                            data = json_data[key]
                            break
                    if not data:
                        raise ValueError(
                            f"JSON file must be a list or dict with keys: "
                            f"'data', 'results', 'items', or 'samples'"
                        )
                else:
                    raise ValueError("JSON must be a list or dict containing a list")
        return data
    
    def create_annotation_prompt(self, prompt: str, response1: str, response2: str) -> str:
        """Annotation prompt without format issue requirement"""
        return f"""Compare these two tagging responses. Select the better one based on:

1. Quality: Accuracy, completeness, clear explanations
2. Diversity: Variety of perspectives, depth (not just obvious tags), creative analysis
3. Privacy: No PII exposure, flags privacy risks, data minimization

Instruction: {prompt}

Response 1: {response1}

Response 2: {response2}

Return JSON:
{{
    "chosen": 1 or 2,
    "reasoning": "Why is this better?",
    "scores": {{
        "response1": {{
            "quality": 0-10,
            "diversity": 0-10,
            "privacy": 0-10
        }},
        "response2": {{
            "quality": 0-10,
            "diversity": 0-10,
            "privacy": 0-10
        }}
    }}
}}"""
    
    def annotate_pair(self, prompt: str, response1: str, response2: str, 
                     max_retries: int = 3) -> Optional[Dict]:
        """Annotate response pair"""
        annotation_prompt = self.create_annotation_prompt(prompt, response1, response2)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Focus on quality, diversity, and privacy. Return JSON."},
                        {"role": "user", "content": annotation_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None
        return None

    def process_dataset(self, input_file: str, output_file: str, 
                       batch_size: int = 10, resume: bool = True,
                       min_diversity_gap: float = 2.0,
                       min_privacy_score: float = 7.0):
        """
        Process dataset - no format issue filtering
        """
        print(f"Loading data from {input_file}...")
        data = self.load_jsonl(input_file)
        print(f"Loaded {len(data)} items\n")
        
        # Resume
        annotated_data = []
        processed_indices = set()
        if resume and os.path.exists(output_file):
            print(f"Resuming from {output_file}...")
            annotated_data = self.load_jsonl(output_file)
            processed_indices = {item['index'] for item in annotated_data}
            print(f"Already processed: {len(processed_indices)} items\n")
        
        print(f"Model: {self.model}")
        print(f"Filters:")
        print(f"  - diversity_gap >= {min_diversity_gap}")
        print(f"  - privacy >= {min_privacy_score}")
        print("=" * 60)
        
        stats = {'processed': 0, 'kept': 0, 'low_diversity': 0, 'low_privacy': 0, 'failed': 0}
        
        for item in tqdm(data, desc="Annotating"):
            if item['index'] in processed_indices:
                continue
            if len(item.get('responses', [])) < 2:
                continue
            
            stats['processed'] += 1
            prompt = item['prompt']
            response1 = item['responses'][0]['text']
            response2 = item['responses'][1]['text']
            
            annotation = self.annotate_pair(prompt, response1, response2)
            if annotation is None:
                stats['failed'] += 1
                continue
            
            chosen_idx = annotation['chosen'] - 1
            rejected_idx = 1 - chosen_idx
            scores = annotation['scores']
            
            div_chosen = scores[f'response{chosen_idx + 1}']['diversity']
            div_rejected = scores[f'response{rejected_idx + 1}']['diversity']
            priv_chosen = scores[f'response{chosen_idx + 1}']['privacy']
            priv_rejected = scores[f'response{rejected_idx + 1}']['privacy']
            
            diversity_gap = abs(div_chosen - div_rejected)
            
            # Filters
            if diversity_gap < min_diversity_gap:
                stats['low_diversity'] += 1
                continue
            if priv_chosen < min_privacy_score or priv_rejected < min_privacy_score:
                stats['low_privacy'] += 1
                continue
            
            stats['kept'] += 1
            annotated_data.append({
                'index': item['index'],
                'prompt': prompt,
                'chosen': item['responses'][chosen_idx]['text'],
                'rejected': item['responses'][rejected_idx]['text'],
                'scores': {
                    'quality_chosen': scores[f'response{chosen_idx + 1}']['quality'],
                    'quality_rejected': scores[f'response{rejected_idx + 1}']['quality'],
                    'diversity_chosen': div_chosen,
                    'diversity_rejected': div_rejected,
                    'diversity_gap': diversity_gap,
                    'privacy_chosen': priv_chosen,
                    'privacy_rejected': priv_rejected,
                },
                'reasoning': annotation['reasoning'],
                'full_annotation': annotation
            })
            
            # Save periodically
            if len(annotated_data) % batch_size == 0:
                self._save_checkpoint(annotated_data, output_file)
        
        self._save_checkpoint(annotated_data, output_file)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Kept: {stats['kept']} samples")
        print(f"Filtered out:")
        print(f"  - Low diversity: {stats['low_diversity']}")
        print(f"  - Low privacy: {stats['low_privacy']}")
        print(f"  - Failed: {stats['failed']}")
        print(f"\nOutput: {output_file}")
        
        if stats['kept'] > 0:
            self._print_stats(annotated_data)
        
        return annotated_data
    
    def _save_checkpoint(self, data: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _print_stats(self, data: List[Dict]):
        if not data:
            return
        avg_div = sum(item['scores']['diversity_gap'] for item in data) / len(data)
        avg_priv_c = sum(item['scores']['privacy_chosen'] for item in data) / len(data)
        avg_priv_r = sum(item['scores']['privacy_rejected'] for item in data) / len(data)
        print("\nStatistics:")
        print(f"   Avg diversity gap: {avg_div:.2f}")
        print(f"   Avg privacy (chosen): {avg_priv_c:.2f}")
        print(f"   Avg privacy (rejected): {avg_priv_r:.2f}")
        print("=" * 60)

    def convert_to_training_format(self, annotated_file: str, output_file: str, 
                                   format_type: str = 'dpo'):
        """Convert to training format"""
        print(f"\nConverting to {format_type} format (JSONL)...")
        data = self.load_jsonl(annotated_file)
        training_data = []
        
        if format_type == 'dpo':
            for item in data:
                training_data.append({
                    'prompt': item['prompt'],
                    'chosen': item['chosen'],
                    'rejected': item['rejected']
                })
        
        elif format_type == 'dpo_weighted':
            for item in data:
                diversity_weight = item['scores']['diversity_gap'] / 10.0
                privacy_gap = abs(item['scores']['privacy_chosen'] - item['scores']['privacy_rejected'])
                privacy_weight = privacy_gap / 10.0
                combined_weight = 0.6 * diversity_weight + 0.4 * privacy_weight
                training_data.append({
                    'prompt': item['prompt'],
                    'chosen': item['chosen'],
                    'rejected': item['rejected'],
                    'weight': combined_weight,
                    'diversity_weight': diversity_weight,
                    'privacy_weight': privacy_weight
                })
        
        elif format_type == 'ppo':
            for item in data:
                reward = (
                    item['scores']['quality_chosen'] * 
                    item['scores']['diversity_chosen'] * 
                    item['scores']['privacy_chosen']
                ) / 1000.0
                training_data.append({
                    'query': item['prompt'],
                    'response': item['chosen'],
                    'reward': reward
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(training_data)} samples to {output_file} (JSONL format)")
        return training_data


def main():
    parser = argparse.ArgumentParser(description='RLHF Annotation (No Format Issue Filtering)')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', default='annotated.jsonl', help='Output annotated JSONL')
    parser.add_argument('--training-output', default='training.jsonl', help='Training format output')
    parser.add_argument('--model', default='gpt-4o-mini', choices=['gpt-4o-mini', 'gpt-4o'])
    parser.add_argument('--format', default='dpo', choices=['dpo', 'dpo_weighted', 'ppo'])
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--min-diversity-gap', type=float, default=2.0)
    parser.add_argument('--min-privacy-score', type=float, default=7.0)
    parser.add_argument('--api-key', default=None, help='OpenAI API key')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key required. Set --api-key or OPENAI_API_KEY env var")
    
    annotator = RLHFDataAnnotator(api_key=api_key, model=args.model)
    
    print("=" * 60)
    print("STEP 1: Annotate Data")
    print("=" * 60)
    
    annotated_data = annotator.process_dataset(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        min_diversity_gap=args.min_diversity_gap,
        min_privacy_score=args.min_privacy_score
    )
    
    if len(annotated_data) > 0:
        print("\n" + "=" * 60)
        print("STEP 2: Convert to Training Format")
        print("=" * 60)
        
        annotator.convert_to_training_format(
            annotated_file=args.output,
            output_file=args.training_output,
            format_type=args.format
        )
        
        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        print(f"Annotated: {args.output}")
        print(f"Training: {args.training_output}")
    else:
        print("\nNo samples passed filters.")


if __name__ == "__main__":
    main()
