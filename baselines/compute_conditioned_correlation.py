import json
import os
import re
from typing import Dict, Any
from collections import defaultdict

from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

from utils.config import Config
from utils.config_constants import LLAVA_OV_MODELS, QWEN2_VL_MODELS
from utils.evaluation import extract_ground_truth_values, save_results_to_latex
from utils.seeds import set_all_seeds


def compute_conditioned_correlation(
    computed_similarities: Dict[Any, float],
    ground_truth_conditioned_similarities: Dict[str, Dict[Any, float]],
    concept: str
    ) -> Dict[str, float]:
    """
    Compute conditioned correlation given computed similarities and ground truth similarities.
    """
    results_dict = defaultdict(dict)

    computed_sim = computed_similarities
    gt_sim = ground_truth_conditioned_similarities[concept]

    computed_scores, gt_scores = [], []

    for pair in computed_sim.keys():
        if pair in gt_sim:
            computed_scores.append(computed_sim[pair])
            gt_scores.append(gt_sim[pair])

    if computed_scores and gt_scores:
        spearman_result = spearmanr(computed_scores, gt_scores)
        kendall_result = kendalltau(computed_scores, gt_scores)

        results_dict = {
            "spearman": round(spearman_result.correlation * 100, 2),
            "kendall": round(kendall_result.correlation * 100, 2),
        }

    return results_dict


def main():

    set_all_seeds(42)
    config = Config()

    gt_sim_dict = extract_ground_truth_values(
        os.path.join(config.general.video_dir, 'ConVIS.jsonl')
    )

    all_models = LLAVA_OV_MODELS | QWEN2_VL_MODELS

    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model in all_models.keys():
        model_dir = os.path.join(config.dataset.conditioned_similarity_dir, 
                                 config.experiment.concept_name, model, 
                                 config.experiment.comparison_approach)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found.")

        if config.experiment.shuffle_actions:
            frames_folder_ends_with = '_frames_actions_shuffled'
        else:
            frames_folder_ends_with = '_frames'
        frame_folders = [d for d in os.listdir(model_dir)
                            if os.path.isdir(os.path.join(model_dir, d)) and d.endswith(frames_folder_ends_with)]
        frame_folders.sort(key=lambda x: int(x.split('_')[0]))

        for frame_folder in frame_folders:
            num_frames = int(frame_folder.split('_')[0])
            sim_path = os.path.join(model_dir, frame_folder, "similarities.json")

            if not os.path.isfile(sim_path):
                raise FileNotFoundError(f"Similarity file {sim_path} not found.")

            with open(sim_path, "r") as f:
                data = json.load(f)

            for entry in data:
                video1 = os.path.basename(entry["video1"])
                video2 = os.path.basename(entry["video2"])
                similarity = entry["similarity"]
                if similarity == 0.0:
                    similarity = 3.0

                pair = (video1, video2)
                all_results[model][num_frames][config.experiment.concept_name][pair] = similarity

    # Compute correlation between computed similarities and ground truth
    results_by_frames = defaultdict(dict)

    for model, frames_dict in all_results.items():
        results_by_frames[model] = {}
        for num_frames, concept_dict in frames_dict.items():
            results_by_frames[model][num_frames] = {}
            results = compute_conditioned_correlation(
                concept_dict[config.experiment.concept_name], gt_sim_dict, config.experiment.concept_name
            )
            results_by_frames[model][num_frames]["spearman"] = results["spearman"]
            results_by_frames[model][num_frames]["kendall"] = results["kendall"]
    # print(results_by_frames)

    headers = ["Model", "Frames", "Spearman (ρ)", "Kendall (τ)"]

    rows = []
    for model in sorted(results_by_frames.keys()):
        for num_frames in sorted(results_by_frames[model].keys()):
            metrics = results_by_frames[model][num_frames]
            spearman = metrics.get("spearman", float('nan'))
            kendall = metrics.get("kendall", float('nan'))
            rows.append([
                model,
                num_frames,
                f"{spearman:.1f}",
                f"{kendall:.2f}"
            ])

    print(f"\033[93mCorrelation metrics computed for all models given the "
          f"\033[95m{config.experiment.comparison_approach} \033[93mcomparison approach:\033[0m")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    latex_output_dir = config.experiment.latex_output_dir
    if latex_output_dir:
        os.makedirs(latex_output_dir, exist_ok=True)
        latex_output_path = os.path.join("conditioned_similarities.tex")
        save_results_to_latex(results_by_frames, latex_output_path)


if __name__ == "__main__":
    main()
