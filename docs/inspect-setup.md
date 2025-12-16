# Inspect Framework Setup

## What is Inspect?

[Inspect](https://inspect.aisi.org.uk/) is UK AISI's open-source eval framework, also used by METR. It handles the boilerplate of running LLM evaluations.

## Architecture

Inspect connects three components:

1. **Datasets** - Collections of samples with inputs and targets
2. **Solvers** - Components that process inputs (e.g., `generate()` calls the model)
3. **Scorers** - Evaluate outputs (text matching, model grading, custom logic)

**Key point:** Inspect handles API calls. You configure the provider, Inspect does the rest.

## Installation

```bash
pip install inspect-ai
```

## Model Providers

Inspect supports many providers. For our work, we use **OpenRouter** which gives unified access to multiple models.

### OpenRouter Setup

```bash
# Install OpenAI package (OpenRouter uses OpenAI-compatible API)
pip install openai

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Run an eval
inspect eval my_task.py --model openrouter/openai/gpt-4o-mini
```

### Model Naming Convention

```
openrouter/<provider>/<model>
```

Examples:
- `openrouter/openai/gpt-4o-mini`
- `openrouter/anthropic/claude-3.5-sonnet`
- `openrouter/meta-llama/llama-3.1-70b-instruct`

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | Defaults to `https://openrouter.ai/api/v1` |

## Writing a Basic Task

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

@task
def my_eval():
    return Task(
        dataset=MemoryDataset([
            Sample(input="Your prompt here", target="Expected behavior")
        ]),
        solver=[generate()],
        scorer=model_graded_fact()
    )
```

## Running Evals

```bash
# Basic run
inspect eval my_task.py --model openrouter/openai/gpt-4o-mini

# View results
inspect view  # Opens web UI with logs
```

## Useful Options

```bash
# Limit samples (for testing)
inspect eval task.py --model ... --limit 5

# Set temperature
inspect eval task.py --model ... -M temperature=0.7

# Multiple models
inspect eval task.py --model openrouter/openai/gpt-4o-mini --model openrouter/anthropic/claude-3.5-sonnet
```

## Resources

- [Inspect Documentation](https://inspect.aisi.org.uk/)
- [Model Providers](https://inspect.aisi.org.uk/providers.html)
- [OpenRouter Models](https://openrouter.ai/models)
