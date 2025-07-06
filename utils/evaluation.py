import os
import json
import ast
import random
from textwrap import dedent

from typing import Dict, Any, List, Union


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

def shuffle_encoded_list(encoded_list: List[Any]) -> Union[List[Any], Any]:
    """
    Attempts to shuffle a list by decoding the first element, which can be a JSON string or a Python literal list string.
    If decoding fails or an error occurs, returns the original list unchanged.
    Ensures the returned list is in a different order when possible.
    """
    try:
        raw = encoded_list[0]
        try:
            inner = json.loads(raw)
        except json.JSONDecodeError:
            inner = ast.literal_eval(raw)

        original = inner[:]
        random.shuffle(inner)

        if inner == original and len(inner) > 1:
            inner = inner[1:] + inner[:1]

        return inner

    except (json.JSONDecodeError, ValueError, SyntaxError, IndexError, TypeError):
        print('Item shuffle failed. Returning original list.')
        return encoded_list

def save_results_to_latex(results_dict: Dict[str, Dict[str, Dict[str, float]]], latex_output_path: str) -> None:
    """
    Save correlation results into a LaTeX table file.
    """
    keys = ['order_of_actions']
    latex = dedent(r"""
        \begin{table}[ht!]
        \def\arraystretch{1.15}
        \centering
        \caption{Baselines for video similarity conditioned on concepts.}
        \scriptsize
        \begin{tabularx}{\textwidth}{
        >{\raggedright\arraybackslash}p{3.2cm}""" +
        ">{\\centering\\arraybackslash}X" * (2 * len(keys)) +
        r"""}
        \toprule
        \textbf{Model} & """ + " & ".join(
            [f"\\multicolumn{{2}}{{c}}{{\\texttt{{Order of Actions}}}}" for _ in keys]
        ) + r" \\" + "\n" + \
        " & " + " & ".join(["$\\rho$ & $\\tau$" for _ in keys]) + r" \\" + "\n" + \
        r"\midrule" + "\n" + \
        rf"\multicolumn{{{1 + 2 * len(keys)}}}{{c}}{{\textit{{VQA-based}}}} \\[-1ex] \\" + "\n"
    )

    for model, values in results_dict.items():
        row = model.replace("_", r"\_") + " & "
        for key in keys:
            row += f"{values[key]['spearman']:.1f} & {values[key]['kendall']:.2f} & "
        latex += row.rstrip(" & ") + r" \\" + "\n"

    latex += r"""\bottomrule
        \end{tabularx}
        \label{tab:semantic-vqa}
        \end{table}
    """

    with open(latex_output_path, "w") as f:
        f.write(latex)
