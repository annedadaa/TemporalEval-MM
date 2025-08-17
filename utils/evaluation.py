import os
import json
import ast
import random
from textwrap import dedent

from typing import Dict, Any, List, Union

from utils.logger import get_logger


logger = get_logger("Evaluation Logger")

def extract_ground_truth_values(ground_truth_jsonl: str) -> Dict[str, Dict[tuple[str, str], Any]]:
    """
    Extracts ground truth similarity values from a JSONL file.
    """
    gt_sim_dict = {'order_of_actions': {}}
    with open(ground_truth_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            v1, v2 = os.path.basename(data['video1']), os.path.basename(data['video2'])
            gt_sim_dict['order_of_actions'][(v1, v2)] = data['OrderOfActions']
    return gt_sim_dict

def shuffle_encoded_list(encoded_list: Any) -> Union[List[str], Any]:
    """
    Returns a shuffled list or the original input if shuffling fails.
    """
    try:
        if len(encoded_list) <= 1:
            return encoded_list

        original = encoded_list[:]
        random.shuffle(encoded_list)

        if encoded_list == original:
            encoded_list = encoded_list[1:] + encoded_list[:1]

        return encoded_list

    except Exception as e:
        logger.warning(f"Shuffle failed. Returning original. Error: {e}")
        return encoded_list
