import torch
import numpy as np
import json
import re
from utils.prompts import SQL_PROMPT, SQL_ANSWER_PROMPT, PRED_PROMPT, SQL_ANSWER_COUNT_PROMPT, REASONING_PROMPT
from models.utils import resize_video


def compute_text_similarity(query_list, key_list, embedding_model, tokenizer, return_all=False):
    encoded_input = tokenizer(
        query_list + key_list,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    device = next(embedding_model.parameters()).device
    encoded_input = {k: v.to(device, non_blocking=True) for k, v in encoded_input.items()}

    with torch.inference_mode():
        model_output = embedding_model(**encoded_input)
        embeddings = model_output.last_hidden_state[:, 0].float()

    query_emb = torch.nn.functional.normalize(embeddings[:len(query_list)], p=2, dim=1)
    key_emb = torch.nn.functional.normalize(embeddings[len(query_list):], p=2, dim=1)
    sims = query_emb @ key_emb.T

    return sims if return_all else torch.mean(sims)
# def compute_text_similarity(query_list, key_list, embedding_model, tokenizer, return_all=False):
#     encoded_input = tokenizer(query_list + key_list, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = embedding_model(**encoded_input)
#         embeddings = model_output[0][:, 0]
#     query_emb = torch.nn.functional.normalize(embeddings[:len(query_list)], p=2, dim=1)
#     key_emb = torch.nn.functional.normalize(embeddings[len(query_list):], p=2, dim=1)
#     sims = query_emb @ key_emb.T
#     if return_all:
#         return sims
#     else:
#         return torch.mean(sims)

def node2indices(node_list, question_type, video_inputs, args):
    n_refine = 8 if question_type == "order" else args.n_refine
    # n_refine = args.n_refine
    sorted_node_list = sorted(map(int, node_list[:n_refine]))
    indices = []
    for idx in sorted_node_list:
        start_time = idx * args.chunk_size
        end_time = min((idx + 1) * args.chunk_size, len(video_inputs[0]))
        indices.extend(range(start_time, end_time))
    indices = set(indices)
    indices = sorted(indices)
    return indices, sorted_node_list

def allocate_node(args, video_graph, entity_graph, query_list, embedding_model, tokenizer, threshold=0.5):
    node_list = []

    for key in list(entity_graph.keys()):
        if compute_text_similarity(query_list, [key], embedding_model, tokenizer) > threshold:
            node_list.extend(entity_graph[key])

    for (node, data) in video_graph.nodes(data=True):
        if node in node_list:
            continue

        key_list = []
        key_list.extend(data.get('entities', []))
        key_list.extend(data.get('actions', []))
        key_list.extend(data.get('scenes', []))
        key_list.extend(data.get('states', []))

        temporal = data.get('temporal', {})
        if isinstance(temporal, dict):
            if temporal.get("stage"):
                key_list.append(str(temporal["stage"]))
            if temporal.get("evidence"):
                key_list.append(str(temporal["evidence"]))

        if data.get("summary"):
            key_list.append(data["summary"])

        if data.get('subtitles') is not None:
            key_list.extend(data.get('subtitles', []))

        if len(key_list) == 0:
            continue

        if compute_text_similarity(query_list, key_list, embedding_model, tokenizer) > threshold:
            node_list.append(node)

    node_list = list(set(node_list))
    return node_list

def extract_choices(question, candidates):
    if "(1)" in question or "(a)" in question:
        pattern = r'\(([a-zA-Z0-9]+)\)\s*(.+?)(?=\s*\([a-zA-Z0-9]+\)|$)'
        matches = re.findall(pattern, question, flags=re.DOTALL)
        query_list = [c[1].strip() for c in matches]
    elif re.search(r"\d+\.", question):
        pattern = r"\d+\.\s+([^\n]+)"
        matches = re.findall(pattern, question)
        query_list = [match.strip() for match in matches]
    elif "-->" in candidates[0]:
        choices = candidates[0].split("-->")
        query_list = [choice.strip() for choice in choices]
    elif len(candidates[0].split(",")) > 2:
        query_list = []
        for candidate in candidates:
            choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
            choices = choices.rstrip('.')
            query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
        query_list = list(set(query_list))
    elif len(candidates[0].split(",")) > 1 and 'and' in candidates[0]:
        query_list = []
        for candidate in candidates:
            choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
            choices = choices.rstrip('.')
            choices = choices.replace(' and ', ',')
            query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
        query_list = list(set(query_list))
    else:
        query_list = candidates
    return query_list


# def count_and_sort_filtered(data):
#     score_dict = {}

#     for index, answers in data.items():
#         if answers is None or not isinstance(answers, dict):
#             continue

#         score = 0
#         for key, value in answers.items():
#             if isinstance(value, str):
#                 v = value.strip().lower()
#                 if v == "yes":
#                     score += 1
#                 elif v not in ["no", "0", ""]:
#                     score += 0.5
#             elif isinstance(value, (int, float)):
#                 if value > 0:
#                     score += float(value)

#         if score > 0:
#             score_dict[index] = score

#     sorted_indices = sorted(score_dict.keys(), key=lambda k: score_dict[k], reverse=True)
#     return score_dict, sorted_indices

# def count_and_sort_filtered(data):
#     """
#     尽量贴近原版：
#     - 对 yes/no 任务：只要该 node 上有非 no/0 的命中，就记一次
#     - 对 numeric 任务：只要该问题回答 > 0，也只把这个问题记作一次命中
#     这样不会因为 numeric 输出 2/3/4 把某个 node 过度放大。
#     """
#     count_dict = {}

#     for index, answers in data.items():
#         if answers is None or not isinstance(answers, dict):
#             continue

#         for _, value in answers.items():
#             if isinstance(value, str):
#                 v = value.strip().lower()
#                 if v != "no" and v != "0" and v != "":
#                     count_dict[index] = count_dict.get(index, 0) + 1
#             elif isinstance(value, (int, float)):
#                 if value > 0:
#                     count_dict[index] = count_dict.get(index, 0) + 1

#     filtered_dict = {k: v for k, v in count_dict.items() if v > 0}
#     sorted_indices = sorted(filtered_dict.keys(), key=lambda k: filtered_dict[k], reverse=True)

#     return filtered_dict, sorted_indices

def _is_malformed_question_echo(text: str) -> bool:
    """
    过滤模型把问题原句/半句直接回显出来的脏输出，例如：
    - "Is there a scene featuring the 'playing trombone' action in the video?"
    - "Is there evidence of the asked action in the video?"
    """
    if not isinstance(text, str):
        return False

    t = text.strip().lower()
    if not t:
        return False

    bad_prefixes = (
        "is there",
        "are there",
        "does ",
        "do ",
        "did ",
        "was ",
        "were ",
        "what ",
        "where ",
        "when ",
        "who ",
        "which ",
        "how many ",
        "is the ",
        "are the ",
        "does the ",
        "do the ",
    )

    if any(t.startswith(p) for p in bad_prefixes):
        return True
    if "?" in t:
        return True
    if "\n" in t:
        return True

    return False


def _is_positive_verification_value(value) -> bool:
    """
    更严格地判断一个 verifier 输出是否算“命中”：
    - "yes" -> 命中
    - 数值 > 0 -> 命中
    - 极短、且不像问题回显的 state 值 -> 命中
    其余一律不算
    """
    if isinstance(value, (int, float)):
        return value > 0

    if not isinstance(value, str):
        return False

    v = value.strip().lower()

    if v in {"", "no", "0", "none", "null"}:
        return False

    if v == "yes":
        return True

    # 过滤把问题原样吐出来的情况
    if _is_malformed_question_echo(v):
        return False

    # 只给极短的 state 文本开口子，避免把长句误判成 positive
    # 例如: "open", "closed", "on table", "in sink"
    if len(v.split()) <= 4:
        return True

    return False


def count_and_sort_filtered(data):
    """
    更稳健的 node 过滤逻辑：
    - 严格 yes 才算 yes/no 命中
    - numeric > 0 才算 numeric 命中
    - 很短且不像问题回显的 state 值才算 state 命中
    - 问题原句回显 / 长句描述 / 非法字符串都不算命中
    """
    count_dict = {}

    for index, answers in data.items():
        if answers is None or not isinstance(answers, dict):
            continue

        hit_count = 0
        for _, value in answers.items():
            if _is_positive_verification_value(value):
                hit_count += 1

        if hit_count > 0:
            count_dict[index] = hit_count

    sorted_indices = sorted(
        count_dict.keys(),
        key=lambda k: count_dict[k],
        reverse=True
    )

    return count_dict, sorted_indices

