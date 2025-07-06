import re
import ast

from typing import Optional, List

SYSPROMPT_TEMPLATE_CONCATENATED = (
    "You are an AI designed to compare two videos based on the visual concept of '{concept_name}'.\n\n"
    "The input consists of a sequence of concatenated frames: the first half represents Video 1, and the second half represents Video 2.\n"
    "Your task is to evaluate how similar these two videos are with respect to the concept '{concept_name}'.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means perfectly similar in terms of '{concept_name}'.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)

SYSPROMPT_TEMPLATE_EXTRACT_ACTIONS = (
    "You are an AI designed to extract actions from videos. \n\n"
    "Analyze the visual content of this video and list the key actions shown, \n"
    "in the order they happen.\n\n"
    "Respond with a Python-style list of short verb-based actions.\n"
    "For example: [\"Enter room\", \"Sit down\", \"Pick up phone\"]\n\n"
    "Keep the list as short and informative as possible — 3 to 6 key steps are enough."
)

SYSPROMPT_TEMPLATE_COMPARE_ACTIONS = (
    "You are an AI designed to compare ordered action sequences.\n\n"
    "The input consists of the two sequences of actions extracted from two videos respectively.\n\n"
    "Video A: {response_1}\n"
    "Video B: {response_2}\n\n"
    "Your task is to evaluate how similar the two action sequences are.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means extremely similar.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)

SYSPROMPT_TEMPLATE_COMPARE_ACTIONS_TO_VIDEO = (
    "You are an AI designed for video understanding and action alignment.\n\n"
    "The input is a list of key actions extracted from a video:\n"
    "{response_1}\n\n"
    "Now, watch the provided video and determine how well it matches the reference sequence "
    "in terms of actions and their order.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means extremely similar.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)

# TODO: revise it
PRE_PROMPT = ("I have a pair of video and a question that I want you to answer. "
              "I need you to strictly follow the format with four specific sections: "
              "SUMMARY, CAPTION, REASONING, and CONCLUSION. "
              "It is crucial that you adhere to this structure exactly as outlined and "
              "that the final answer in the CONCLUSION matches the standard correct answer precisely. "
              "To explain further: In SUMMARY, briefly explain what steps you'll take to solve the problem. "
              "In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. "
              "In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. "
              "In CONCLUSION, give the final answer in a direct format, and it must match the correct answer exactly. "
              "Here's how the format should look: <SUMMARY> [Summarize how you will approach the problem and explain "
              "the steps you will take to reach the answer.] </SUMMARY> <CAPTION> [Provide a detailed description of "
              "the image, particularly emphasizing the aspects related to the question.] </CAPTION> <REASONING> "
              "[Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.] "
              "</REASONING> <CONCLUSION> [State the final answer in a clear and direct format. "
              "It must match the correct answer exactly.] </CONCLUSION> (Do not forget </CONCLUSION>!) "
              "Please apply this format meticulously to analyze the given image and answer the related question, "
              "ensuring that the answer matches the standard one perfectly.\nQuestion: ")

# TODO: revise it
CAPTIONING_SYSPROMPT = (
            "You are an AI designed to generate structured and detailed descriptions of videos. "
            "Your task is to produce coherent paragraphs where each paragraph corresponds to a distinct scene or segment within the video. "
            "For every segment, describe:\n"
            "- The primary actions or events unfolding.\n"
            "- The location and environmental context in which these events occur.\n"
            "- The main characters or subjects and their roles.\n"
            "- Key objects and their interactions within the scene.\n\n"
            "Ensure that each paragraph clearly separates different scenes or shifts in activity, while maintaining a continuous, flowing narrative throughout the video. "
            "Aim for descriptions that are clear, complete, and logically organized."
)


def get_prompt(
    concept_name: Optional[str] = None,
    comparison_approach: str = "concatenate",
    use_zs_cot: bool = False,
    use_llava_cot: bool = False,
    stage_extract_actions: bool = False,
    extracted_actions: List[List[str]] = []
    ) -> str:
    """
    Returns a prompt for the given concept and comparison approach.
    """

    prompt = ""

    if comparison_approach == "concatenate":
        prompt = SYSPROMPT_TEMPLATE_CONCATENATED.format(concept_name=re.sub(r'(?<!^)(?=[A-Z])',
                                               ' ', concept_name))

    elif comparison_approach == "extract_compare":

        if stage_extract_actions:
            prompt = SYSPROMPT_TEMPLATE_EXTRACT_ACTIONS
        else:
            response_1 = safe_parse_action_list(extracted_actions[0][0])
            response_2 = safe_parse_action_list(extracted_actions[1][0])

            prompt = SYSPROMPT_TEMPLATE_COMPARE_ACTIONS.format(response_1=response_1, response_2=response_2)

    elif comparison_approach == "bidirectional":
        if stage_extract_actions:
            prompt = SYSPROMPT_TEMPLATE_EXTRACT_ACTIONS
        else:
            response = safe_parse_action_list(extracted_actions[0][0])
            prompt = SYSPROMPT_TEMPLATE_COMPARE_ACTIONS_TO_VIDEO.format(response_1=response)

    if use_llava_cot:
        prompt = PRE_PROMPT + prompt
    elif use_zs_cot:
        prompt = prompt + "Think step by step."

    return prompt


def safe_parse_action_list(string) -> list:
    """
    Parse a string into a list of actions.
    """
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        s_fixed = string.strip()
        if not s_fixed.endswith(']'):
            s_fixed += ']'
        if s_fixed.count('"') % 2 != 0:
            s_fixed += '"'
        try:
            return ast.literal_eval(s_fixed)
        except Exception:
            return re.findall(r'"(.*?)"', s_fixed)
