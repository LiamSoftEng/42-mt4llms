"""
Decodes private test cases from the LiveCodeBench dataset.
Private tests are stored as base64 -> zlib -> pickle -> JSON string.
"""

import base64
import zlib
import pickle
import json


def decode_private_test_cases(raw: str) -> list:
    """
    Decode private test cases into a list of {input, output, testtype} dicts.
    Returns [] if decoding fails.
    """
    if not raw or not raw.strip():
        return []
    try:
        compressed = base64.b64decode(raw)
        decompressed = zlib.decompress(compressed)
        json_str = pickle.loads(decompressed)
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except Exception as e:
        print(f"  [decode] Failed: {e}")
    return []