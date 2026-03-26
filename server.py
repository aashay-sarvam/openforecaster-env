"""
OpenForecaster ORS Environment Server

Loads nikhilchandak/OpenForesight from HuggingFace and serves open-ended
forecasting questions as an OpenReward Standard (ORS) environment.

Reward — LLM-as-a-judge (cloud API, no local GPU required):
  Primary  : Anthropic claude-haiku-4-5 via secrets["ANTHROPIC_API_KEY"]
  Fallback : OpenAI gpt-4o-mini       via secrets["OPENAI_API_KEY"]
  Heuristic: numeric relative-error + substring match (if no API key set)

The judge prompt handles paraphrases, specificity, and numeric relative-error
(<= 1%), matching the evaluation criteria from the OpenForecaster paper.

Tool: submit_answer(answer: str, confidence: float)
"""

import logging
import re
import os
from pydantic import BaseModel, Field
from datasets import load_dataset

from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_TOOL_SUFFIX = (
    "\n\nWhen you have finished your reasoning, call "
    "submit_answer(answer=<your concise answer>, confidence=<P(correct) in [0,1]>) "
    "to submit your final answer and end the episode."
)

# ---------------------------------------------------------------------------
# Judge prompt — verbatim from local_judge/llm_judge.py → get_judge_prompt_with_gt
# ---------------------------------------------------------------------------

def _build_judge_prompt(question: str, ground_truth: str, response: str) -> str:
    return (
        "Your task is to judge whether the given response to a question matches a given "
        "ground truth answer or not. You are provided with a question, a ground truth "
        "response, and the response you need to judge.\n"
        'For a response to "match", it must have the same information as in the ground-truth '
        "(not less nor unnecessary extra).\n"
        "The response can be more specific than the ground-truth (for example, \"Labrador\" is "
        "more specific than \"dog\"), or have additional possible correct answers. But it must "
        "cover everything mentioned in the ground-truth. It is okay if it covers it in "
        "different words, i.e. paraphrased.\n"
        "For numeric answers, the relative error, defined as |response - ground truth| / "
        "mean(response, ground truth), must be <= 1% for the response to be judged as a "
        "correct match. Here, if the ground truth is a specific numeric quantity but the "
        "response is a range, then they don't match (even if the range contains the ground truth).\n\n"
        "Possible judgments:\n\n"
        '"0": The response does not match the ground-truth answer.\n'
        '"1": The response matches the ground-truth.\n\n'
        f'Question: "{question}"\n'
        f'Ground truth: "{ground_truth}"\n'
        f'Response: "{response}"\n\n'
        "Your job is to ONLY check whether the given response matches the ground truth answer "
        "or not in the context of the question. You DO NOT NEED to assess the correctness of "
        "the response. This is part of an automated evaluation process, therefore you MUST "
        "OUTPUT your final answer as \"0\" or \"1\" in <judgment> tags.\n"
        "Think step by step and end your response with 0 OR 1 <judgment> TAGS."
    )


def _parse_judgment(text: str) -> float | None:
    """Extract 0 or 1 from <judgment>X</judgment> tags."""
    m = re.search(r"<judgment>\s*([01])\s*</judgment>", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # fallback: last standalone 0 or 1 in the text
    nums = re.findall(r"\b([01])\b", text)
    if nums:
        return float(nums[-1])
    return None

# ---------------------------------------------------------------------------
# Heuristic fallback reward (substring + numeric relative-error)
# ---------------------------------------------------------------------------

def _extract_number(s: str) -> float | None:
    """Pull the first number (possibly with commas/units) out of a string."""
    s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    return float(m.group()) if m else None


def _heuristic_reward(ground_truth: str, model_answer: str) -> float:
    """
    1. Try numeric relative-error match (<= 1%).
    2. Fall back to case-insensitive substring containment.
    """
    gt = ground_truth.strip()
    ans = model_answer.strip()

    gt_num = _extract_number(gt)
    ans_num = _extract_number(ans)
    if gt_num is not None and ans_num is not None:
        denom = (abs(gt_num) + abs(ans_num)) / 2
        if denom == 0:
            return 1.0 if gt_num == ans_num else 0.0
        rel_err = abs(gt_num - ans_num) / denom
        return 1.0 if rel_err <= 0.01 else 0.0

    # substring match
    gt_l = gt.lower()
    ans_l = ans.lower()
    return 1.0 if (gt_l in ans_l or ans_l in gt_l) else 0.0

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _llm_judge(
    question: str,
    ground_truth: str,
    model_answer: str,
    secrets: dict,
) -> float | None:
    """
    Call a cheap LLM to judge the answer.  Returns 0.0 or 1.0, or None on failure.
    Tries Anthropic (claude-haiku-4) then OpenAI (gpt-4o-mini).
    """
    prompt = _build_judge_prompt(question, ground_truth, model_answer)

    # --- Anthropic ---
    api_key = secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            judgment = _parse_judgment(text)
            if judgment is not None:
                return judgment
            logger.warning("Anthropic judge returned unparseable response: %s", text[:200])
        except Exception as exc:
            logger.warning("Anthropic judge failed: %s", exc)

    # --- OpenAI ---
    api_key = secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            judgment = _parse_judgment(text)
            if judgment is not None:
                return judgment
            logger.warning("OpenAI judge returned unparseable response: %s", text[:200])
        except Exception as exc:
            logger.warning("OpenAI judge failed: %s", exc)

    return None  # no API available / both failed

# ---------------------------------------------------------------------------
# Dataset loading (once at import time)
# ---------------------------------------------------------------------------

def _load_split(split_name: str) -> list[dict]:
    logger.info(f"Loading OpenForesight split={split_name!r} …")
    ds = load_dataset("nikhilchandak/OpenForesight", split=split_name)
    tasks = []
    for i, ex in enumerate(ds):
        tasks.append({
            "id": f"{split_name}-{i}",
            "question_title": ex.get("question_title") or ex.get("question", ""),
            "prompt_text": (ex.get("prompt_without_retrieval") or "").strip(),
            "answer": str(ex.get("answer", "")).strip(),
            "answer_type": str(ex.get("answer_type", "")).strip(),
            "resolution_date": str(ex.get("resolution_date", "")),
        })
    logger.info(f"Loaded {len(tasks)} tasks (split={split_name!r})")
    return tasks


try:
    _TRAIN = _load_split("train")
    _VALIDATION = _load_split("validation")
    _TEST = _load_split("test")
except Exception as exc:
    logger.error(f"Failed to load OpenForesight: {exc}")
    _TRAIN = _VALIDATION = _TEST = []

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    id: str
    question_title: str
    prompt_text: str
    answer: str
    answer_type: str
    resolution_date: str


class SubmitAnswerParams(BaseModel):
    answer: str = Field(
        description=(
            "Your concise final answer (a few words: a name, number, date, "
            "organisation, etc.)."
        )
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Your confidence that this answer is correct, as a probability in [0, 1].",
    )

# ---------------------------------------------------------------------------
# ORS Environment
# ---------------------------------------------------------------------------

class OpenForecaster(Environment):
    """
    Open-ended forecasting questions from OpenForesight.

    Splits : train (52 k tasks), validation (207 tasks), test (302 tasks).
    Tool   : submit_answer(answer, confidence)
    Reward : LLM-as-a-judge (same prompt as local_judge/llm_judge.py), with
             heuristic fallback when no API key is configured.

    Secrets (set in OpenReward environment settings):
      ANTHROPIC_API_KEY  — preferred judge (claude-haiku-4)
      OPENAI_API_KEY     — alternative judge (gpt-4o-mini)
    """

    def __init__(self, task_spec: JSONObject = {}, secrets: dict = {}):
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self._secrets = secrets

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        mapping = {"train": _TRAIN, "validation": _VALIDATION, "test": _TEST}
        if split not in mapping:
            raise ValueError(
                f"Unknown split {split!r}. Choose from: {list(mapping)}"
            )
        return mapping[split]

    @classmethod
    def list_splits(cls):
        return [
            Split(name="train",      type="train"),
            Split(name="validation", type="validation"),
            Split(name="test",       type="test"),
        ]

    def get_prompt(self) -> list[TextBlock]:
        text = self.config.prompt_text + _TOOL_SUFFIX
        return [TextBlock(type="text", text=text)]

    @tool
    def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Submit your final answer to the forecasting question.
        Provide a concise answer (name, number, date, …) and your confidence (0–1).
        Calling this tool ends the episode.
        """
        cfg = self.config
        model_answer = params.answer.strip()

        # Try LLM judge first (same logic as local_judge/llm_judge.py)
        reward = _llm_judge(
            question=cfg.question_title,
            ground_truth=cfg.answer,
            model_answer=model_answer,
            secrets=self._secrets,
        )
        judge_used = "llm"

        if reward is None:
            # Fallback: heuristic (numeric relative-error + substring)
            reward = _heuristic_reward(cfg.answer, model_answer)
            judge_used = "heuristic"
            logger.warning(
                "Using heuristic reward for question %r (no LLM judge available)",
                cfg.question_title[:60],
            )

        correct = reward > 0.5
        feedback = (
            f"Ground truth: {cfg.answer!r}. "
            f"Your answer: {model_answer!r}. "
            f"{'Correct!' if correct else 'Incorrect.'} "
            f"Reward: {reward:.2f} (judge: {judge_used})"
        )
        return ToolOutput(
            blocks=[TextBlock(type="text", text=feedback)],
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    Server([OpenForecaster]).run()
