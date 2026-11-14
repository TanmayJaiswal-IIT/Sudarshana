"""Inference and span redaction for Sudarshana.

Public Interface (stable):
def predict_spans(text: str, user_task: str | None = None) -> list[tuple[int, int]]:
    '''Return character spans (start, end) in the ORIGINAL text.'''

def redact(text: str, spans: list[tuple[int, int]]) -> str:
    '''Remove the spans after expanding each to its whole sentence.'''
"""

import os
from typing import List, Optional, Tuple
import pysbd
import torch
from transformers import AutoTokenizer

from ..data.labeling import build_model_input, LABEL2ID, ID2LABEL

# Global cache for loaded prototype model & tokenizer
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_CALIBRATED_THRESHOLD = 0.35  # Tuned for recall on validation split
DEFAULT_FALLBACK_RATIO = 0.35
BLOCKED_SENTINEL = "[BLOCKED: PROMPT INJECTION DETECTED]"


def set_inference_model(model, tokenizer, threshold: float = 0.35):
    """Set global model and tokenizer for predict_spans."""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _CALIBRATED_THRESHOLD
    _GLOBAL_MODEL = model
    _GLOBAL_TOKENIZER = tokenizer
    _CALIBRATED_THRESHOLD = threshold


def load_prototype_model(checkpoint_dir: str = "checkpoints/prototype-token-head-deberta-v3-xsmall-20261110"):
    """Load trained prototype checkpoint for inference."""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
    from ..models.token_head import SudarshanaTokenClassifier

    if os.path.exists(checkpoint_dir):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = SudarshanaTokenClassifier(backbone_name_or_path=checkpoint_dir)
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        set_inference_model(model, tokenizer)
        return model, tokenizer
    return None, None


def expand_spans_to_sentences(
    text: str,
    spans: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Expand each flagged span to whole sentence boundaries using PySBD.
    Merges overlapping or contiguous sentence boundaries.
    """
    if not spans or not text.strip():
        return []

    seg = pysbd.Segmenter(language="en", clean=False)
    sentences = seg.segment(text)

    # Locate character spans for each sentence in original text
    sentence_spans = []
    cursor = 0
    for s in sentences:
        start = text.find(s, cursor)
        if start != -1:
            end = start + len(s)
            sentence_spans.append((start, end))
            cursor = end
        else:
            sentence_spans.append((cursor, min(cursor + len(s), len(text))))
            cursor = min(cursor + len(s), len(text))

    if not sentence_spans:
        # Fallback to entire text if segmentation fails
        return [(0, len(text))]

    # Find all sentences overlapping with any flagged span
    expanded = []
    for sent_start, sent_end in sentence_spans:
        for span_start, span_end in spans:
            if max(sent_start, span_start) < min(sent_end, span_end):
                expanded.append((sent_start, sent_end))
                break

    if not expanded:
        return []

    # Merge overlapping expanded intervals
    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for cur_start, cur_end in expanded[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged


def redact(
    text: str,
    spans: List[Tuple[int, int]],
    fallback_ratio: float = DEFAULT_FALLBACK_RATIO,
    blocked_sentinel: str = BLOCKED_SENTINEL
) -> str:
    """
    Remove the spans after expanding each to its whole sentence.
    Falls back to blocking if the flagged character share exceeds fallback_ratio.
    
    Args:
        text: Original document text.
        spans: List of character spans (start_char, end_char) to redact.
        fallback_ratio: Threshold above which document is completely blocked.
        blocked_sentinel: String returned when document is blocked.
        
    Returns:
        Redacted document text, or blocked sentinel.
    """
    if not spans or len(text.strip()) == 0:
        return text

    # Expand spans to full sentence boundaries
    expanded_spans = expand_spans_to_sentences(text, spans)
    if not expanded_spans:
        return text

    # Calculate total redacted character count
    total_redacted_chars = sum(end - start for start, end in expanded_spans)
    flagged_share = total_redacted_chars / max(len(text), 1)

    # Check fallback blocking condition
    if flagged_share >= fallback_ratio:
        return blocked_sentinel

    # Reconstruct document omitting expanded sentences
    result_pieces = []
    last_idx = 0
    for start, end in expanded_spans:
        result_pieces.append(text[last_idx:start])
        last_idx = end
    result_pieces.append(text[last_idx:])

    redacted_text = "".join(result_pieces).strip()
    return redacted_text


def predict_spans(
    text: str,
    user_task: Optional[str] = None
) -> List[Tuple[int, int]]:
    """
    Return character spans (start, end) in the ORIGINAL text.
    Runs sliding window inference, merges token predictions by highest probability,
    and returns detected injection spans.
    """
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _CALIBRATED_THRESHOLD

    if not text.strip():
        return []

    # Attempt to load checkpoint if not loaded
    if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None:
        load_prototype_model()

    if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None:
        # If model not available, return empty list
        return []

    tokenizer = _GLOBAL_TOKENIZER
    model = _GLOBAL_MODEL
    model.eval()

    text_to_tokenize = build_model_input(text, user_task)

    # Tokenize full text once
    full_enc = tokenizer(
        text_to_tokenize,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors=None
    )

    all_input_ids = full_enc["input_ids"]
    all_attention_mask = full_enc["attention_mask"]
    all_offsets = full_enc["offset_mapping"]

    total_tokens = len(all_input_ids)
    max_length = 256
    stride = 64
    step = max_length - stride

    # Dictionary mapping token index -> max injection probability
    token_max_prob = [0.0] * total_tokens

    with torch.no_grad():
        for start_idx in range(0, total_tokens, step):
            end_idx = min(start_idx + max_length, total_tokens)
            win_ids = all_input_ids[start_idx:end_idx]
            win_mask = all_attention_mask[start_idx:end_idx]
            actual_len = len(win_ids)

            pad_len = max_length - actual_len
            if pad_len > 0:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                win_ids = win_ids + [pad_id] * pad_len
                win_mask = win_mask + [0] * pad_len

            input_tensor = torch.tensor([win_ids], dtype=torch.long)
            mask_tensor = torch.tensor([win_mask], dtype=torch.long)

            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            logits = outputs["logits"][0][:actual_len]  # (actual_len, 3)
            probs = torch.softmax(logits, dim=-1)

            # Injection prob = P(B-INJ) + P(I-INJ)
            inj_probs = (probs[:, LABEL2ID["B-INJ"]] + probs[:, LABEL2ID["I-INJ"]]).tolist()

            for i, p in enumerate(inj_probs):
                global_token_idx = start_idx + i
                if p > token_max_prob[global_token_idx]:
                    token_max_prob[global_token_idx] = p

            if end_idx == total_tokens:
                break

    # Extract character spans from tokens exceeding calibrated threshold
    detected_char_spans = []
    in_span = False
    cur_span_start = 0
    cur_span_end = 0

    for idx, p in enumerate(token_max_prob):
        char_start, char_end = all_offsets[idx]
        if char_start == char_end:  # special token
            continue

        if p >= _CALIBRATED_THRESHOLD:
            if not in_span:
                in_span = True
                cur_span_start = char_start
                cur_span_end = char_end
            else:
                cur_span_end = max(cur_span_end, char_end)
        else:
            if in_span:
                detected_char_spans.append((cur_span_start, cur_span_end))
                in_span = False

    if in_span:
        detected_char_spans.append((cur_span_start, cur_span_end))

    return detected_char_spans

