import json
import hashlib
from typing import Any

def normalize(obj: Any, float_precision: int | None = None):
    if isinstance(obj, dict):
        return {k: normalize(v, float_precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v, float_precision) for v in obj]
    if isinstance(obj, float) and float_precision is not None:
        return round(obj, float_precision)
    return obj

def canonical_json_bytes(obj: Any, float_precision: int | None = None) -> bytes:
    norm = normalize(obj, float_precision)
    s = json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return s.encode('utf-8')

def file_hash(path: str, float_precision: int | None = None) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    b = canonical_json_bytes(obj, float_precision)
    return hashlib.sha256(b).hexdigest()

def compare_files(path_a: str, path_b: str, float_precision: int | None = None) -> bool:
    return file_hash(path_a, float_precision) == file_hash(path_b, float_precision)

if __name__ == '__main__':
    import sys
    a, b = sys.argv[1], sys.argv[2]
    equal = compare_files(a, b, float_precision=None)  # set precision if desired
    print('IDENTICAL' if equal else 'DIFFER')