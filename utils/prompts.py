import re
import ast

from typing import Optional, List, Any

SYSPROMPT_TEMPLATE_CONCATENATED = (
    "You are an AI designed to compare two videos based on the visual concept of order of actions.\n\n"
    "The input consists of a sequence of concatenated frames: the first half represents Video 1, and the second half represents Video 2.\n"
    "Your task is to evaluate how similar these two videos are with respect to their order of actions.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means perfectly similar in terms of order of actions.\n"
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
    "Your task is to evaluate how similar the two action sequences are in terms of their order.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means extremely similar.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)

SYSPROMPT_TEMPLATE_COMPARE_ACTIONS_TO_VIDEO = (
    "You are an AI designed for video understanding and action alignment.\n\n"
    "The input is a list of key actions extracted from a video:\n"
    "{response_1}\n\n"
    "Now, watch the provided video and determine how well it matches the reference sequence "
    "in terms of their order of actions.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means extremely similar.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)

# Adapted from https://arxiv.org/pdf/2411.10440
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
 


def get_prompt(
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
        prompt = SYSPROMPT_TEMPLATE_CONCATENATED

    elif comparison_approach == "extract_compare":

        if stage_extract_actions:
            prompt = SYSPROMPT_TEMPLATE_EXTRACT_ACTIONS
        else:
            response_1 = safe_parse_action_list(extracted_actions[0])
            response_2 = safe_parse_action_list(extracted_actions[1])

            prompt = SYSPROMPT_TEMPLATE_COMPARE_ACTIONS.format(response_1=response_1, response_2=response_2)

    elif comparison_approach == "bidirectional":
        if stage_extract_actions:
            prompt = SYSPROMPT_TEMPLATE_EXTRACT_ACTIONS
        else:
            response = safe_parse_action_list(extracted_actions[0])
            prompt = SYSPROMPT_TEMPLATE_COMPARE_ACTIONS_TO_VIDEO.format(response_1=response)

    if use_llava_cot:
        prompt = PRE_PROMPT + prompt
    elif use_zs_cot:
        prompt = prompt + "Think step by step."

    return prompt

def safe_parse_action_list(value: Any) -> List[str]:
    """
    Safely parse a value into a list of actions.
    Handles:
    - Actual lists
    - Strings containing list-like content (even inside larger text)
    - Malformed list strings
    - Markdown/code block formatted list strings
    """
    if isinstance(value, list):
        # If it's already a list of strings, return it
        if all(isinstance(v, str) for v in value):
            return value
        # If it's a list with one string that looks like a list
        elif len(value) == 1 and isinstance(value[0], str):
            value = value[0]
        else:
            return []

    if not isinstance(value, str):
        return []

    value = value.strip()

    # Try extracting the first list-like structure from the string
    match = re.search(r'\[.*?\]', value, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
        candidate = value

    # Try parsing with ast.literal_eval
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass

    # Fallback: extract quoted strings
    return re.findall(r'"(.*?)"', candidate)
