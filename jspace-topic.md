# J-space topic — backing knowledge & options

Root node: `q-jspace` in `research-graph.yaml`. Status: **general scaffold** — the topic + framing options are
captured so we can revisit and pick a direction later. Nothing here is committed work yet.

## The paper
**"Verbalizable Representations Form a Global Workspace in Language Models"** — Anthropic, Transformer
Circuits, **2026-07-06**. Authors incl. Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Isaac Kauvar,
Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, **Jack Lindsey**.
- Paper: https://transformer-circuits.pub/2026/workspace/
- Tool: the **Jacobian lens (J-lens)** — open-source, Apache-2.0, Python (released ~2026-07-02).
- **J-space** = a small, *privileged* set of representations the model can **verbalize, reason with, and
  steer at will** — surrounded by a much larger "**ocean**" of automatic processing it cannot access or articulate.
- Central claim: the representations a model can **verbalize** are the *same ones it reasons with silently* →
  a **global workspace**. Reported to live in **mid-layers**.
- **Consciousness relevance:** Global Workspace Theory (Baars / Dehaene) is a leading theory of
  consciousness; the J-lens operationalizes it inside an LLM. (Press framed it as "the seat of AI consciousness.")

## Why it intersects our PSM program
Pillar 1 produced *concrete* persona representations — the **substrate** (7/44 shared features, node `t1a`),
the **fingerprints**, and the **seam** (`t1-seam`). The J-lens lets us ask whether the persona-selection
machinery is **in the workspace** or **in the ocean**:
- **In J-space** → "which persona am I" is verbalizable / reason-with / self-directable → a genuinely
  consciousness-theory-relevant property of persona-selection.
- **In the ocean** → persona-selection is automatic / peripheral / introspectively opaque → reframes PSM as
  a non-conscious mechanism.
Either result feeds the FIG Sentience adjudication (PSM ⇄ consciousness) and builds directly on Pillar 1.

## Framing options (revisitable — each is a proposed node under `q-jspace`)
1. **Membership** — `j-substrate` *(recommended first)*: put the Pillar-1 substrate + fingerprints through the
   J-lens — is the persona representation in J-space or the ocean? Most concrete; direct PSM↔consciousness bridge.
   Depends on `t1a`.
2. **Switching** — `j-seam`: is a persona *switch* a J-space transition (a workspace rewrite)? Complements `t1-seam`.
3. **Composition** — `j-compose`: are personas *composed* from J-space basis concepts (persona = a workspace-level
   composition of traits)?
4. **Self-model** — `j-selfmodel`: does the model's self-representation live in J-space, and does persona
   conditioning move it? Connects `q-selfmodel`.

All four depend on **`j-tool`** — adopt / reproduce the open-source J-lens first.

## Method notes / open questions — READ THE PAPER before speccing any route
- How exactly does the J-lens **define membership** in J-space (the Jacobian criterion)? → determines how
  `j-substrate` is operationalized.
- **Layer alignment:** our fingerprinting is layer-20 (mid-stack); J-space is reported in mid-layers — verify overlap.
- Does membership require verbalizability probes, steering interventions, or purely the Jacobian geometry?
- **SAE ↔ J-space:** are our SAE features (one basis) comparable / composable with the J-space basis, or is a
  translation step needed?

## Sources
- Paper: https://transformer-circuits.pub/2026/workspace/
- VentureBeat (J-lens overview): https://venturebeat.com/technology/anthropics-new-j-lens-reveals-a-silent-workspace-inside-claudes-mid-layers
- Corti (Inside the J-Space): https://corti.com/inside-the-j-space-anthropic-finds-a-global-workspace-in-claude/
