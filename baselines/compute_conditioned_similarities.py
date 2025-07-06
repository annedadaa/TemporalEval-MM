import json
import os
import re
import warnings

from typing import Dict, List, Literal, Union

import torch
from tqdm import tqdm

from models.llava import LLaVAOneVisionModel
from models.qwen import Qwen2VLModel

from utils.config import Config
from utils.config_constants import LLAVA_OV_MODELS, QWEN2_VL_MODELS
from utils.evaluation import shuffle_encoded_list
from utils.image_utils import visualize_stacked_frames
from utils.prompts import get_prompt
from utils.seeds import set_all_seeds
from utils.video_io import (
    encode_concat_video_llava,
    encode_concat_video_qwen,
    encode_single_video_llava,
    encode_single_video_qwen,
    read_video_pairs,
)


warnings.filterwarnings("ignore", category=UserWarning)


def load_model(model_version: str, device: Union[str, torch.device]) -> Union[LLaVAOneVisionModel, Qwen2VLModel]:
    """
    Load the Video-LLM model based on the model path.
    """

    if model_version in LLAVA_OV_MODELS.keys():

        model_vqa = LLaVAOneVisionModel(model_name=model_version, device=device)
        model_vqa.model.eval()

    elif model_version in QWEN2_VL_MODELS.keys():

        model_vqa = Qwen2VLModel(model_name=model_version, device=device)
        model_vqa.model.eval()

    else:
        raise NotImplementedError(f"Model {model_version} currently not implemented")
    return model_vqa


def parse_answer_to_score(answer: str) -> float:
    """
    Parse the answer from the model to a score.
    """
    answer = answer.split("\n")[0]
    try:
        score = float(answer)

    except ValueError:
        print(f"Could not parse answer: {answer}")
        score = 0.0
    return score


def compute_similarity(
    video_pairs: List[Dict[str, str]],
    model_vqa: Union[LLaVAOneVisionModel, Qwen2VLModel],
    config: Config
    ) -> List[Dict[str, Union[str, float]]]:
    """
    Compute similarity scores for pairs of videos using a Video-LLM model.
    """

    video_pairs_sim = []

    for video_pair in tqdm(video_pairs):


        if config.experiment.comparison_approach == "concatenate":

            if model_vqa.model_name in ["LLaVA-OneVision-Qwen2-7B"]:
                concat_video_frames = encode_concat_video_llava(
                    config, model_vqa, video_pair["video1"], video_pair["video2"])

            elif model_vqa.model_name in ["Qwen2.5-VL-7B-Instruct"]:
                concat_video_frames_raw = encode_concat_video_qwen(
                    config, model_vqa, video_pair["video1"], video_pair["video2"])
                concat_video_frames = [{"type": "video", "video": concat_video_frames_raw}]
            
            if config.experiment.visualize_frames:
                filename_1 = os.path.splitext(os.path.basename(video_pair['video1']))[0]
                filename_2 = os.path.splitext(os.path.basename(video_pair['video2']))[0]

                if isinstance(concat_video_frames[0], dict):
                    video_frames = concat_video_frames[0]["video"]
                else:
                    video_frames = concat_video_frames

                visualize_stacked_frames(video_frames, 
                os.path.join(config.experiment.visualized_frames_dir, 
                             model_vqa.model_name, 
                             config.experiment.comparison_approach), 
                f"{filename_1}_{filename_2}_{config.experiment.max_frames_num}_frames.jpg")


            prompt = get_prompt(concept_name=config.experiment.concept_name,
                                comparison_approach=config.experiment.comparison_approach)

            # print(f"\033[94mUsing System Prompt(s)\n\n{prompt}\033[0m")

            answer = model_vqa.generate(
                data=concat_video_frames,
                max_new_tokens=config.model.max_new_tokens,
                conditioned_system_prompt=prompt,
                )

            print('PROCESSED ANSWER:', answer)

        elif config.experiment.comparison_approach in ["extract_compare", "bidirectional"]:

            if model_vqa.model_name in ["LLaVA-OneVision-Qwen2-7B"]:
                video_1_frames = encode_single_video_llava(config, model_vqa, video_pair["video1"])
                video_2_frames = encode_single_video_llava(config, model_vqa, video_pair["video2"])

            elif model_vqa.model_name in ["Qwen2.5-VL-7B-Instruct"]:
                video_1_frames = encode_single_video_qwen(config, model_vqa, video_pair["video1"])
                video_2_frames = encode_single_video_qwen(config, model_vqa, video_pair["video2"])

                video_1_frames = [{"type": "video", "video": video_1_frames}]
                video_2_frames = [{"type": "video", "video": video_2_frames}]

            if config.experiment.visualize_frames:
                if isinstance(video_1_frames[0], dict) and isinstance(video_2_frames[0], dict):
                    video_frames_1 = video_1_frames[0]["video"]
                    video_frames_2 = video_2_frames[0]["video"]
                else:
                    video_frames_1 = video_1_frames
                    video_frames_2 = video_2_frames

                visualize_stacked_frames(video_frames_1, 
                os.path.join(config.experiment.visualized_frames_dir, 
                             model_vqa.model_name, 
                             config.experiment.comparison_approach), 
                f"{os.path.splitext(os.path.basename(video_pair['video1']))[0]}_{config.experiment.max_frames_num}_frames.jpg")

                visualize_stacked_frames(video_frames_2, 
                os.path.join(config.experiment.visualized_frames_dir, 
                             model_vqa.model_name, 
                             config.experiment.comparison_approach), 
                f"{os.path.splitext(os.path.basename(video_pair['video2']))[0]}_{config.experiment.max_frames_num}_frames.jpg")

            prompt_extract = get_prompt(concept_name=config.experiment.concept_name,
                                        comparison_approach=config.experiment.comparison_approach,
                                        stage_extract_actions=True)

            print(f"\033[94mUsing System Prompt(s)\n\n{prompt_extract}\033[0m")

            video_1_caption = model_vqa.generate(
                data=video_1_frames,
                max_new_tokens=config.model.max_new_tokens_caption,
                conditioned_system_prompt=prompt_extract,
            )

            video_2_caption = model_vqa.generate(
                data=video_2_frames,
                max_new_tokens=config.model.max_new_tokens_caption,
                conditioned_system_prompt=prompt_extract,
            )

            print(f"\033[95mVIDEO CAPTION 1:\n\n{video_1_caption}\033[0m")
            # print(f"\033[93mVIDEO CAPTION 2:\n\n{video_2_caption}\033[0m")

            if config.experiment.shuffle_actions:
                video_1_caption = shuffle_encoded_list(video_1_caption)

                print(f"\033[95mSHUFFLED VIDEO CAPTION 1:\n\n{video_1_caption}\033[0m")


            if config.experiment.comparison_approach == "extract_compare":

                prompt_compare = get_prompt(concept_name=config.experiment.concept_name,
                                            comparison_approach=config.experiment.comparison_approach,
                                            extracted_actions=[video_1_caption, video_2_caption])

                # print(f"\033[94mUsing System Prompt(s)\n\n{prompt_compare}\033[0m")

                answer = model_vqa.generate(
                    data=None,
                    max_new_tokens=config.model.max_new_tokens,
                    conditioned_system_prompt=prompt_compare,
                )
                print('PROCESSED ANSWER:', answer)

            if config.experiment.comparison_approach == "bidirectional":

                # print("VIDEO CAPTION 1:", video_1_caption)
                prompt_compare_1 = get_prompt(concept_name=config.experiment.concept_name,
                                              comparison_approach=config.experiment.comparison_approach,
                                              extracted_actions=[video_1_caption])

                # print(f"\033[94mUsing System Prompt(s)\n\n{prompt_compare_1}\033[0m")

                answer_1 = model_vqa.generate(
                    data=video_2_frames,
                    max_new_tokens=config.model.max_new_tokens,
                    conditioned_system_prompt=prompt_compare_1,
                )


                prompt_compare_2 = get_prompt(concept_name=config.experiment.concept_name,
                                              comparison_approach=config.experiment.comparison_approach,
                                              extracted_actions=[video_2_caption])

                # print(f"\033[94mUsing System Prompt(s)\n\n{prompt_compare_2}\033[0m")

                answer_2 = model_vqa.generate(
                    data=video_1_frames,
                    max_new_tokens=config.model.max_new_tokens,
                    conditioned_system_prompt=prompt_compare_2,
                )

                # Compute the average similarity score
                answer = [str((int(answer_1[0]) + int(answer_2[0])) / 2)]
                print('PROCESSED ANSWER:', answer)

        else:
            raise ValueError(f"Invalid video comparison approach: {config.experiment.comparison_approach}")


        sim = parse_answer_to_score(answer[0])


        video_pairs_sim.append(
            {
                "video1": video_pair["video1"],
                "video2": video_pair["video2"],
                "similarity": sim,
            }
        )

    return video_pairs_sim


def main():

    set_all_seeds(42)
    config = Config()

    print(f"\033[93mComputing similarities with model: {config.model.vqa_model}\033[0m")
    video_pairs = read_video_pairs(config.dataset.pairs_jsonl)

    if config.experiment.shuffle_actions:
        shuffle_actions_flag = "_actions_shuffled"
    else:
        shuffle_actions_flag = ""

    if config.experiment.comparison_approach:
        output_dir = os.path.join(config.dataset.conditioned_similarity_dir,
                                  config.experiment.concept_name,
                                  config.model.vqa_model,
                                  config.experiment.comparison_approach,
                                  f"{str(config.experiment.max_frames_num)}_frames{shuffle_actions_flag}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "similarities.json")
    else:
        raise ValueError(f"Specify the video comparison approach.")

    print(f"\033[92mOutput file: {output_file}\033[0m")

    model_vqa = load_model(config.model.vqa_model, config.general.device)

    print(f"\033[92mConcept: {re.sub(r'(?<!^)(?=[A-Z])', ' ', config.experiment.concept_name)}\033[0m")
    print(f"\033[92mComparison Approach: {re.sub(r'(?<!^)(?=[A-Z])', ' ', config.experiment.comparison_approach)}\033[0m")
    print(f"\033[92mModel: {config.model.vqa_model}\033[0m")

    if config.model.vqa_model in ['LLaVA-OneVision-Qwen2-7B', 'Qwen2.5-VL-7B-Instruct']:
        video_pairs_sim = compute_similarity(video_pairs, model_vqa, config)

    else:
        raise ValueError(f"Model {config.model.vqa_model} currently not implemented")

    with open(output_file, "w") as f:
        json.dump(video_pairs_sim, f, indent=4)
    print(f"\033[92mSimilarities saved to: {output_file}\033[0m")


if __name__ == "__main__":
    main()
