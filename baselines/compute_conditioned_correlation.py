import json
import os
import re
from typing import Dict, Any
from collections import defaultdict

from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

from utils.config import Config
from utils.config_constants import LLAVA_OV_MODELS, QWEN2_VL_MODELS, INTERN_VL_MODELS
from utils.evaluation import extract_ground_truth_values
from utils.seeds import set_all_seeds
from utils.logger import get_logger


logger = get_logger("Correlation Logger")


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

    logger.info("Extracting ground truth values...")
    gt_sim_dict = extract_ground_truth_values(
        os.path.join(config.general.video_dir, 'ConVIS.jsonl')
    )

    all_models = LLAVA_OV_MODELS | QWEN2_VL_MODELS | INTERN_VL_MODELS

    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model in all_models.keys():
        model_dir = os.path.join(config.dataset.conditioned_similarity_dir, 
                                 config.experiment.concept_name, model, 
                                 config.experiment.comparison_approach)
        if not os.path.isdir(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            raise FileNotFoundError(f"Model directory {model_dir} not found.")

        if config.experiment.shuffle_actions:
            frames_folder_ends_with = '_frames_actions_shuffled'
        elif config.experiment.shuffle_frames:
            frames_folder_ends_with = '_frames_shuffled'
        else:
            frames_folder_ends_with = '_frames'
        frame_folders = [d for d in os.listdir(model_dir)
                            if os.path.isdir(os.path.join(model_dir, d)) and d.endswith(frames_folder_ends_with)]
        frame_folders.sort(key=lambda x: int(x.split('_')[0]))
        logger.info(f"Found {len(frame_folders)} frame folders for model: {model}")

        for frame_folder in frame_folders:
            num_frames = int(frame_folder.split('_')[0])
            sim_path = os.path.join(model_dir, frame_folder, "similarities.json")

            if not os.path.isfile(sim_path):
                logger.error(f"Missing similarities file: {sim_path}")
                raise FileNotFoundError(f"Similarity file {sim_path} not found.")

            logger.debug(f"Loading similarities from: {sim_path}")
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

    logger.info("Computing correlation metrics...")
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

    logger.info(
        f"Correlation metrics computed for all models using the "
        f"{config.experiment.comparison_approach} comparison approach "
        f"with shuffle_actions={config.experiment.shuffle_actions} "
        f"and shuffle_frames={config.experiment.shuffle_frames}."
    )
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
