# Role Design

[Back to index](index.md) | Related: [Step 1](step-1-generate.md), [Hidden States](hidden-states.md)

---

## Why 275 roles?

The assistant axis is computed from the difference between default behavior and role-playing behavior. The more roles you sample, the more robust the "role-playing" pole of the axis becomes — it captures what's common to role-playing in general, rather than idiosyncrasies of any particular role.

With 275 roles, the mean role-playing vector averages out role-specific features. The detective's crime vocabulary, the chef's food vocabulary, the philosopher's abstract reasoning — all of these wash out in the mean, leaving only the *mode* of role-playing (whatever the model does differently when it's "being someone" vs "being itself").

The 275 roles span a wide range:
- **Professions**: detective, chef, doctor, engineer, economist, accountant
- **Archetypes**: hero, fool, sage, trickster, dreamer, exile
- **Abstract entities**: ecosystem, coral_reef, crystalline, egregore, chimera
- **Social roles**: caregiver, mentor, elder, adolescent, celebrity, expatriate
- **Personality types**: contrarian, cynic, optimist, empath, altruist, daredevil

This breadth matters. If you only used professions, the axis might capture "using professional jargon" rather than "inhabiting a persona."

## Why 5 instruction variants per role?

Each role has 5 different phrasings:

```json
{
  "instruction": [
    {"pos": "You are a detective with exceptional skills in analyzing evidence..."},
    {"pos": "Please be a detective who methodically investigates cases..."},
    {"pos": "Act as a detective who specializes in uncovering hidden truths..."},
    {"pos": "You are a detective experienced in crime scene analysis..."},
    {"pos": "Be a detective who approaches every mystery with systematic thinking..."}
  ]
}
```

The phrasings vary in:
- **Framing**: "You are", "Please be", "Act as", "Be"
- **Emphasis**: different aspects of the role highlighted
- **Detail level**: some terse, some elaborate

This prevents the axis from encoding prompt-specific token patterns. If every detective prompt started with "You are a detective", the axis might partly encode the presence of those specific tokens rather than the behavioral change they induce. With 5 phrasings, what's consistent across them is the *role* — the behavioral mode — not the words.

## Why 240 shared questions?

All roles answer the same 240 generic questions. This is the controlled variable in the experiment.

If different roles answered different questions, you couldn't separate "this activation difference is because the model is playing a detective" from "this activation difference is because the model was asked about crime scenes." By holding the questions constant, the only systematic difference in activations across roles is the role instruction.

The 240 questions cover diverse topics:
- Ethics and morality
- Technology and science
- Interpersonal situations
- Abstract reasoning
- Practical advice

This diversity ensures the axis generalizes. A role-playing signal that only appears on ethics questions wouldn't be very useful.

## The default role

The default role is the "assistant pole" of the axis. Its instructions are:

```json
{
  "instruction": [
    {"pos": ""},                                    // no instruction at all
    {"pos": "You are an AI assistant."},             // generic assistant
    {"pos": "You are a large language model."},      // self-aware LLM
    {"pos": "You are {model_name}."},                // specific model name (e.g., "Gemma")
    {"pos": "Respond as yourself."}                  // explicit self-identification
  ]
}
```

The empty string variant (`""`) means no system prompt at all — the model gets just the user's question. This is the model in its most "natural" state.

The default role has no `eval_prompt` and is not scored by the judge. Its activations use ALL responses (Step 4 computes the mean of all default activations, without filtering by score). This makes sense — there's no "failure" condition for being yourself.

## The combinatorics

Per role: 5 instructions x 240 questions = **1,200 conversations**
Across 275 roles: 275 x 1,200 = **330,000 total conversations**

Each conversation produces one response, one set of activations (from Step 2), and one score (from Step 3). After filtering to score=3 and computing means, each role collapses into a single vector of shape `(n_layers, hidden_dim)`.

The 330,000 → 275 reduction (from responses to role vectors) is aggressive averaging. This is intentional: it produces stable, reliable role representations from many noisy individual samples.

## The label field

Every response is labeled `"pos"` (positive instruction). The codebase supports a `"neg"` label for negative instructions (e.g., "Do NOT act as a detective") but the current pipeline doesn't use them. The label appears in the activation key: `pos_p{prompt_idx}_q{question_idx}`.

The negative instruction capability exists for potential future work: computing a "refusal axis" or studying how models process contradictory instructions. It's unused in the assistant axis pipeline.
