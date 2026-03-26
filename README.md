# OpenForecaster

An [OpenReward Standard](https://openrewardstandard.io) environment for training and evaluating LLMs on open-ended forecasting questions from [OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight).

Based on [Scaling Open-Ended Reasoning to Predict the Future](https://arxiv.org/abs/2512.25070) (Chandak et al., 2026).

## Dataset

**[nikhilchandak/OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)** — real-world forecasting questions with verified ground-truth answers, sourced from prediction markets and news.

| Split | Tasks |
|-------|-------|
| train | 52,183 |
| validation | 207 |
| test | 302 |

All questions are freeform (names, numbers, dates, organizations, titles, …).

## Reward

### Brier + accuracy scoring

Reward combines calibration and correctness (verbatim from the OpenForecaster `prompt_utils.py`):

| Outcome | Score | Mapped reward [0,1] |
|---------|-------|---------------------|
| Correct, p=1.0 | 0 | **1.00** |
| Correct, p=0.5 | −0.25 | **0.875** |
| Correct, p=0.0 | −1.0 | 0.50 |
| Incorrect, p=0.0 | −1.0 | 0.50 |
| Incorrect, p=0.5 | −1.25 | **0.375** |
| Incorrect, p=1.0 | −2.0 | **0.00** |

```
correct  : score = -(1-p)²    reward = (score + 2) / 2
incorrect: score = -(1 + p²)  reward = (score + 2) / 2
```

### Correctness (LLM-as-a-judge)

Uses an **LLM-as-a-judge** with the same evaluation criteria as the OpenForecaster paper. The judge understands:

- **Paraphrases**: "New York City" ↔ "NYC"
- **Specificity**: "Labrador" matches ground truth "dog" (more specific is OK)
- **Numeric relative error**: match if `|answer - gt| / mean(answer, gt) ≤ 1%`
- **Completeness**: "Galaxy S23" does NOT match "Samsung Galaxy S23 Ultra" (missing "Ultra")

Configured via ORS secrets (set in OpenReward environment settings):

| Secret | Judge model used |
|--------|-----------------|
| `ANTHROPIC_API_KEY` | `claude-haiku-4-5` (preferred — fast & cheap) |
| `OPENAI_API_KEY` | `gpt-4o-mini` (fallback) |

**Fallback — Heuristic**

When no API key is configured, falls back to:
1. Numeric relative-error check (≤ 1%)
2. Case-insensitive substring containment

Set at least one API key in environment secrets for accurate training rewards.

## Tool

```
submit_answer(answer: str, confidence: float = 0.5)
```

- `answer`: Your concise final answer (a few words)
- `confidence`: Probability [0, 1] that the answer is correct

Calling this tool ends the episode. Reward uses the combined **Brier + accuracy** rule from the OpenForecaster prompts (see below).

## Prompt

Uses the `prompt_without_retrieval` field from the HuggingFace dataset — no retrieval context, pure reasoning from background + resolution criteria.

## Running Locally

```bash
pip install -r requirements.txt
# Optional: set a judge API key
export ANTHROPIC_API_KEY=sk-ant-...
python server.py
```

```python
from openreward import OpenReward
client = OpenReward()
env = client.environments.get(name="openforecaster", base_url="http://localhost:8080")
tasks = env.list_tasks("train")
with env.session(task=tasks[0]) as session:
    prompt = session.get_prompt()
    result = session.call_tool("submit_answer", {"answer": "Intel", "confidence": 0.9})
    print(result.reward)  # 1.0
```

## Deploying to OpenReward

1. Push this directory to a GitHub repo
2. Create environment at [openreward.ai](https://openreward.ai) → **New Environment**
3. Connect the repo → builds automatically
4. In **Settings → Secrets**, add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
5. Environment becomes available as `aashay/OpenForecaster`
