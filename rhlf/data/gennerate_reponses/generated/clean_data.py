import json
import re
from typing import List, Dict, Any

def normalize_tag(tag: str) -> str:
    if not isinstance(tag, str):
        return tag
    tag = tag.replace("_", " ").strip()
    tag = re.sub(r'(?<!^)(?=[A-Z])', ' ', tag)
    tag = re.sub(r'\s*-\s*', '-', tag)
    tag = re.sub(r'\s*/\s*', '/', tag)
    tag = re.sub(r'\s+', ' ', tag)
    return tag.strip()

def parse_text_field(text: str):
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 去掉 ```json 包裹和多余符号
        cleaned = text.strip().strip("`")
        cleaned = re.sub(r"^json", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        return json.loads(cleaned)

def clean_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for entry in data:
        for resp in entry.get("responses", []):
            t = resp.get("text")
            if isinstance(t, str):
                try:
                    arr = parse_text_field(t)
              
                    if isinstance(arr, list):
                        for item in arr:
                            if isinstance(item, dict) and "tag" in item:
                                item["tag"] = normalize_tag(item["tag"])
            
                    resp["text"] = arr
                except Exception as e:
                    resp["text_error"] = f"{type(e).__name__}: {e}"
    return data

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python clean_data.py <input.json> <output.json>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = clean_dataset(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Done. Output written to: {out_path}")
