# Contacts: Lu et al. Assistant Axis Team

Paper: "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models" (arXiv:2601.10387, Jan 2026)

## Authors

| Name | Affiliation | Notes |
|------|-------------|-------|
| **Christina Lu** | MATS, Anthropic Fellows, Oxford | First/corresponding author. Owns HuggingFace dataset (`lu-christina`). Primary contact for conversation datasets. |
| **Jack Gallagher** | Anthropic | |
| **Jonathan Michala** | MATS | Other MATS-affiliated author. |
| **Kyle Fish** | Anthropic | |
| **Jack Lindsey** | Anthropic | Senior/supervising author, co-corresponding. |

## Acknowledged contributors

| Name | Role |
|------|------|
| Cem Anil | Early mentorship on the research |
| Ethan Perez | Supporting MATS/Anthropic Fellows programs |
| Avery Griffin | Supporting the programs |
| Henry Sleight | Supporting the programs |
| John Hughes | Technical workflow assistance and compute management |
| Clement Dumas | Feedback on drafts |
| Egg Syntax | Feedback on drafts |
| Josh Batson | Feedback on drafts |

## What to ask for

- The full multi-turn conversation transcripts (~400+ conversations across 4 domains, 3 auditor models)
- The 20 handwritten personas and 400 generated topics
- The 15,000 user message embeddings and regression data
- Any unpublished analysis on metacognitive prompt effects specifically
- **Conversation generation script**: The published repo has transcripts and analysis code but not the script that orchestrates the auditor-target turn-taking loop. The repo contains `auditor.json` (prompt variants) and sample transcripts (showing Sonnet 4.5 as auditor, Llama 70B as target), but no code that actually runs the multi-turn conversations — alternating turns, managing context, handling `<END_CONVERSATION>` termination, saving transcripts. Was this done with a separate internal script? A notebook? An existing framework? Would they be willing to share it or describe the setup?
