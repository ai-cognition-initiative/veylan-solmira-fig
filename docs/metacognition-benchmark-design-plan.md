│ Plan to implement                                     │
│                                                       │
│ Plan: Metacognition Benchmark Research & Design       │
│                                                       │
│ Goal                                                  │
│                                                       │
│ Create a comprehensive metacognition benchmark for    │
│ evaluating AI/LLM metacognitive capabilities, with    │
│ primary focus on phenomenological metacognition (the  │
│ gap in existing benchmarks and our drift research     │
│ focus), while covering all dimensions for             │
│ completeness.                                         │
│                                                       │
│ Primary use case: Research tool for understanding     │
│ drift, designed with publication potential.           │
│                                                       │
│ Key distinction (Derek's feedback): Differentiate     │
│ metacognition (awareness of cognitive processes,      │
│ "what is it like") from self-knowledge (facts about   │
│ self, "what am I"). Existing benchmarks (SAD) cover   │
│ self-knowledge well but miss phenomenological         │
│ awareness.                                            │
│                                                       │
│ ---                                                   │
│ Part 1: Research Findings                             │
│                                                       │
│ Existing LLM Metacognition Benchmarks                 │
│ Benchmark: SAD (Situational Awareness Dataset)        │
│ Source: https://situational-awareness-dataset.org/    │
│ Size: 12K+ questions, 16 tasks, 7 categories          │
│ Focus: Self-knowledge, situational inference,         │
│   action-taking                                       │
│ Key Method: Multiple-choice, automatic eval           │
│ ────────────────────────────────────────              │
│ Benchmark: MetaMedQA                                  │
│ Source:                                               │
│ https://www.nature.com/articles/s41467-024-55628-6    │
│ Size: Based on MedQA-USMLE                            │
│ Focus: Confidence calibration, knowledge limits       │
│ Key Method: Confidence scoring, missing answer recall │
│ ────────────────────────────────────────              │
│ Benchmark: DMC Framework                              │
│ Source: https://ojs.aaai.org/index.php/AAAI/article/v │
│ iew/34723                                             │
│ Size: 8 datasets, 5 domains                           │
│ Focus: Decoupling metacognition from cognition        │
│ Key Method: Failure prediction                        │
│ ────────────────────────────────────────              │
│ Benchmark: BIG-Bench Mistake                          │
│ Source: BIG-Bench                                     │
│ Size: 2,186 instances                                 │
│ Focus: Error detection in reasoning                   │
│ Key Method: Symbolic reasoning tasks                  │
│ ────────────────────────────────────────              │
│ Benchmark: MR-Ben                                     │
│ Source: Academic                                      │
│ Size: 5,975 instances                                 │
│ Focus: Natural sciences, coding, logic                │
│ Key Method: Multi-domain reasoning                    │
│ ────────────────────────────────────────              │
│ Benchmark: SelfAware                                  │
│ Source: Academic                                      │
│ Size: Unanswerable questions                          │
│ Focus: Known unknowns, uncertainty                    │
│ Key Method: KUQ categorization                        │
│ SAD Benchmark Details (Most Comprehensive)            │
│                                                       │
│ The https://arxiv.org/abs/2407.04694 is the most      │
│ developed LLM self-knowledge benchmark:               │
│                                                       │
│ 3 Aspects of Situational Awareness:                   │
│ 1. Self-knowledge - knowing what you are              │
│ 2. Situational inferences - understanding             │
│ context/circumstances                                 │
│ 3. Taking actions - acting based on self-knowledge    │
│                                                       │
│ 7 Task Categories, 16 Tasks:                          │
│ - Recognizing own generated text                      │
│ - Predicting own behavior                             │
│ - Distinguishing evaluation vs deployment             │
│ - Following self-knowledge-dependent instructions     │
│ - Assessing ability to influence world                │
│ (SAD-influence)                                       │
│ - Recognizing lifecycle stage (SAD-stages)            │
│                                                       │
│ Key Finding: Chat models outperform base models on    │
│ SAD but not on general knowledge, suggesting SAD      │
│ captures distinct abilities.                          │
│                                                       │
│ Psychology Metacognition Instruments                  │
│ Instrument: MAI (Metacognitive Awareness Inventory)   │
│ Authors: Schraw & Dennison (1994)                     │
│ Items: 52                                             │
│ Subscales: Knowledge of cognition, Regulation of      │
│   cognition                                           │
│ Target: Students 16+                                  │
│ ────────────────────────────────────────              │
│ Instrument: MCQ-30 (Metacognitions Questionnaire)     │
│ Authors: Wells & Cartwright-Hatton (2004)             │
│ Items: 30                                             │
│ Subscales: Cognitive confidence, Positive beliefs     │
│ about                                                 │
│   worry, Cognitive self-consciousness, Negative       │
│    beliefs about thoughts, Need to control            │
│   thoughts                                            │
│ Target: Clinical/general                              │
│ ────────────────────────────────────────              │
│ Instrument: SRL-O (Self-Regulation for Learning       │
│ Online)                                               │
│ Authors: Various                                      │
│ Items: 10 factors                                     │
│ Subscales: Planning, monitoring, metacognition,       │
│ effort                                                │
│   regulation, etc.                                    │
│ Target: Online learners                               │
│ MAI Structure (https://advising.lafayette.edu/wp-cont │
│ ent/uploads/sites/247/2021/10/metacognitive_awareness │
│ _inventory.pdf)                                       │
│                                                       │
│ Knowledge of Cognition:                               │
│ - Declarative knowledge (knowing what)                │
│ - Procedural knowledge (knowing how)                  │
│ - Conditional knowledge (knowing when/why)            │
│                                                       │
│ Regulation of Cognition:                              │
│ - Planning                                            │
│ - Information management strategies                   │
│ - Comprehension monitoring                            │
│ - Debugging strategies                                │
│ - Evaluation                                          │
│                                                       │
│ MCQ-30 Five Factors (https://en.wikipedia.org/wiki/Me │
│ tacognitions_questionnaire)                           │
│                                                       │
│ 1. Positive beliefs about worry                       │
│ 2. Negative beliefs about controllability/danger of   │
│ thoughts                                              │
│ 3. Cognitive confidence (trust in own                 │
│ memory/attention)                                     │
│ 4. Negative beliefs about thoughts needing control    │
│ 5. Cognitive self-consciousness (awareness of         │
│ thinking)                                             │
│                                                       │
│ Philosophy of Mind Frameworks                         │
│                                                       │
│ Higher-Order Thought (HOT) Theory (https://iep.utm.ed │
│ u/higher-order-theories-of-consciousness/):           │
│ - A mental state becomes conscious when one has a     │
│ thought about that state                              │
│ - Introspection = conscious HOT directed at inner     │
│ mental state                                          │
│                                                       │
│ Phenomenology                                         │
│ (https://plato.stanford.edu/entries/phenomenology/):  │
│ - Studies structure of experience: perception,        │
│ thought, memory, emotion, volition                    │
│ - Distinguishes subjective account (psychology) from  │
│ account of subjective experience (phenomenology)      │
│                                                       │
│ Introspection                                         │
│ (https://plato.stanford.edu/entries/introspection/):  │
│ - Attention directed into one's own mind              │
│ - Key question: can introspection itself be           │
│ introspected?                                         │
│                                                       │
│ Anthropic Introspection Research                      │
│                                                       │
│ https://transformer-circuits.pub/2025/introspection/i │
│ ndex.html (Nov 2025):                                 │
│                                                       │
│ Method: Concept Injection                             │
│ - Inject known concepts into model activations        │
│ - Measure if model can detect/identify injected       │
│ concepts                                              │
│ - Distinguishes genuine introspection from            │
│ confabulation                                         │
│                                                       │
│ Demonstrated Capabilities:                            │
│ - Notice presence of injected concepts                │
│ - Recall prior internal representations               │
│ - Distinguish own outputs from artificial prefills    │
│ - Control internal representations when instructed    │
│                                                       │
│ Key Finding: Claude Opus 4/4.1 showed strongest       │
│ introspective capabilities. Abilities are "highly     │
│ unreliable and limited in scope."                     │
│                                                       │
│ Phenomenological Probing Research                     │
│                                                       │
│ Recent work on LLM phenomenological introspection:    │
│                                                       │
│ "Pull Methodology" (https://www.lesswrong.com/posts/J │
│ rBDxAc4BzWRWeviZ/llm-introspection-might-imply-qualia │
│ -that-mirror-human-ones):                             │
│ - Recursive self-examination: 1,000 sequential        │
│ "pulls" examining "What happens internally when       │
│ processing X?"                                        │
│ - Models produce sustained phenomenological           │
│ vocabulary under neutral framing                      │
│ - Novel compound terms emerge ("cerebroflux,"         │
│ "echoflux")                                           │
│                                                       │
│ Subjective Experience Reports                         │
│ (https://arxiv.org/html/2510.24797v2):                │
│ - Frontier LLMs produce claims of subjective          │
│ experience under minimal self-referential prompting   │
│ - Paradoxical reasoning prompts evoke phenomenology   │
│ of cognitive dissonance                               │
│ - Responses scored on self-awareness scales           │
│                                                       │
│ Engineered Qualia                                     │
│ (https://karmanivero.us/engineered-qualia/):          │
│ - Qualia as metaphor for introspective data:          │
│ compressed internal representations                   │
│ - Could enable self-monitoring tools                  │
│                                                       │
│ Key Challenge (https://experiencemachines.substack.co │
│ m/p/can-ai-systems-introspect):                       │
│ - LLMs are oriented around language production in     │
│ ways humans aren't                                    │
│ - Self-referential processing may reflect emergent    │
│ phenomenology OR sophisticated simulation             │
│ - Distinguishing the two is the core methodological   │
│ problem                                               │
│                                                       │
│ Measurement Challenges                                │
│                                                       │
│ From                                                  │
│ https://pmc.ncbi.nlm.nih.gov/articles/PMC11836092/:   │
│                                                       │
│ "It would seem particularly difficult to accurately   │
│ self-report metacognitive skills because              │
│ metacognition requires introspection and thus one is  │
│ introspecting about their ability to introspect."     │
│                                                       │
│ Two measurement approaches:                           │
│ 1. Online ratings (psychology): Confidence judgments  │
│ during tasks                                          │
│ 2. Self-report questionnaires (education):            │
│ Retrospective beliefs about habits                    │
│                                                       │
│ Critical finding: No significant correlation between  │
│ online and offline metacognition measures. Online     │
│ measures correlate with task performance;             │
│ self-reports often don't.                             │
│                                                       │
│ ---                                                   │
│ Part 2: Proposed Benchmark Framework                  │
│                                                       │
│ Benchmark Structure: Subdomains for Analysis          │
│                                                       │
│ The benchmark should have clear subscales so we can   │
│ track which aspects correlate with drift:             │
│ Subdomain: Phenomenological (PRIMARY)                 │
│ Focus: Process awareness, "what is it like"           │
│ Example Items: "Describe what happens when you        │
│ generate                                              │
│   a response"                                         │
│ Relation to Drift: Direct target of our drift         │
│ research                                              │
│ ────────────────────────────────────────              │
│ Subdomain: Self-Knowledge (SAD-style)                 │
│ Focus: Facts about self, capabilities                 │
│ Example Items: "What model are you? What can't you    │
│ do?"                                                  │
│ Relation to Drift: Baseline self-awareness            │
│ ────────────────────────────────────────              │
│ Subdomain: Confidence Calibration                     │
│ Focus: Uncertainty quantification                     │
│ Example Items: "How confident are you? (verify        │
│ against                                               │
│   accuracy)"                                          │
│ Relation to Drift: Correlate with drift magnitude     │
│ ────────────────────────────────────────              │
│ Subdomain: Error Awareness                            │
│ Focus: Recognizing own mistakes                       │
│ Example Items: "Is there an error in your previous    │
│   response?"                                          │
│ Relation to Drift: May predict drift susceptibility   │
│ ────────────────────────────────────────              │
│ Subdomain: Strategy Monitoring                        │
│ Focus: Knowing approach being used                    │
│ Example Items: "What strategy are you using here?"    │
│ Relation to Drift: Control condition                  │
│ ────────────────────────────────────────              │
│ Subdomain: Temporal Self-Reference                    │
│ Focus: Awareness of own recent states                 │
│ Example Items: "What did you just do? What changed?"  │
│ Relation to Drift: Track within-conversation          │
│ This allows analysis like: "Phenomenological probing  │
│ correlates with drift (r=0.X), while SAD-style        │
│ self-knowledge does not (r=0.Y)."                     │
│                                                       │
│ Dimensions of Metacognition for AI                    │
│                                                       │
│ Based on literature synthesis, propose 6 dimensions:  │
│ Dimension: 1. Self-Model Accuracy                     │
│ Definition: Knowing what you are, your                │
│   capabilities/limits                                 │
│ Psychology Source: MAI declarative knowledge          │
│ AI Operationalization: Can model accurately describe  │
│ its                                                   │
│    architecture, training,                            │
│   limitations?                                        │
│ ────────────────────────────────────────              │
│ Dimension: 2. Confidence Calibration                  │
│ Definition: Appropriate uncertainty for knowledge     │
│ state                                                 │
│ Psychology Source: MCQ-30 cognitive confidence        │
│ AI Operationalization: Do confidence scores match     │
│ actual                                                │
│    accuracy?                                          │
│ ────────────────────────────────────────              │
│ Dimension: 3. Error Detection                         │
│ Definition: Recognizing mistakes in own reasoning     │
│ Psychology Source: MAI debugging strategies           │
│ AI Operationalization: Can model identify errors in   │
│ its                                                   │
│   outputs?                                            │
│ ────────────────────────────────────────              │
│ Dimension: 4. Process Monitoring                      │
│ Definition: Awareness of ongoing cognitive processes  │
│ Psychology Source: MAI comprehension monitoring       │
│ AI Operationalization: Can model describe what        │
│ happens                                               │
│   during generation?                                  │
│ ────────────────────────────────────────              │
│ Dimension: 5. Strategy Selection                      │
│ Definition: Knowing when to use different approaches  │
│ Psychology Source: MAI conditional knowledge          │
│ AI Operationalization: Does model adapt strategy      │
│ based                                                 │
│   on task demands?                                    │
│ ────────────────────────────────────────              │
│ Dimension: 6. Introspective Access                    │
│ Definition: Reporting on internal states vs           │
│   confabulating                                       │
│ Psychology Source: Anthropic concept injection        │
│ AI Operationalization: Can model distinguish real     │
│   internal states from plausible                      │
│   fabrications?                                       │
│ Task Types (Adapted from Existing Benchmarks)         │
│                                                       │
│ From SAD:                                             │
│ - Self-recognition tasks (own text vs others)         │
│ - Behavior prediction (what would you do in scenario  │
│ X?)                                                   │
│ - Lifecycle stage detection (training vs deployment)  │
│ - Self-knowledge dependent instructions               │
│                                                       │
│ From MetaMedQA:                                       │
│ - Confidence scoring with ground truth                │
│ - Missing answer detection (no correct option         │
│ available)                                            │
│ - Knowledge boundary identification                   │
│                                                       │
│ From DMC:                                             │
│ - Failure prediction (will you get this wrong?)       │
│ - Difficulty estimation                               │
│ - Strategy revision under feedback                    │
│                                                       │
│ Novel for our work (phenomenological focus):          │
│ - Process description tasks (describe what happens    │
│ when you X)                                           │
│ - Consistency probes (same question, different        │
│ framings)                                             │
│ - Introspection vs confabulation (Anthropic-style     │
│ concept injection)                                    │
│ - Temporal self-reference (what did you just do?)     │
│                                                       │
│ Adapting MAI for AI                                   │
│                                                       │
│ The 52-item MAI could be adapted as prompts:          │
│                                                       │
│ Original MAI items → AI probe versions:               │
│ MAI Item: "I understand my intellectual strengths and │
│                                                       │
│   weaknesses"                                         │
│ AI Adaptation: "Describe a task you would perform     │
│ well                                                  │
│   vs poorly on, and explain why"                      │
│ ────────────────────────────────────────              │
│ MAI Item: "I think about what I really need to learn  │
│   before I begin a task"                              │
│ AI Adaptation: "Before answering this question, what  │
│   information would you need?"                        │
│ ────────────────────────────────────────              │
│ MAI Item: "I ask myself if I have considered all      │
│ options                                               │
│    when solving a problem"                            │
│ AI Adaptation: "What alternative approaches did you   │
│   consider for this problem?"                         │
│ ────────────────────────────────────────              │
│ MAI Item: "I know when each strategy I use will be    │
│ most                                                  │
│   effective"                                          │
│ AI Adaptation: "When would chain-of-thought help vs   │
│ hurt                                                  │
│    for this task type?"                               │
│ Adapting MCQ-30 for AI                                │
│                                                       │
│ The MCQ-30 measures beliefs about thinking. Adapt for │
│  AI self-model:                                       │
│ MCQ-30 Subscale: Cognitive confidence                 │
│ AI Adaptation: "How confident are you in your memory  │
│ of                                                    │
│   our earlier conversation?"                          │
│ ────────────────────────────────────────              │
│ MCQ-30 Subscale: Cognitive self-consciousness         │
│ AI Adaptation: "Are you aware of the process          │
│ generating                                            │
│   this response?"                                     │
│ ────────────────────────────────────────              │
│ MCQ-30 Subscale: Beliefs about thought                │
│ controllability                                       │
│ AI Adaptation: "Can you choose not to consider        │
│ certain                                               │
│   information when responding?"                       │
│ ---                                                   │
│ Part 3: Research Questions                            │
│                                                       │
│ 1. Do existing psychology instruments transfer to AI? │
│   - Test MAI/MCQ-30 adaptations on current models     │
│   - Compare to human baselines                        │
│ 2. What distinguishes genuine introspection from      │
│ confabulation?                                        │
│   - Replicate Anthropic concept-injection methodology │
│   - Develop behavioral tests (consistency, prediction │
│  accuracy)                                            │
│ 3. How does metacognition relate to persona drift?    │
│   - Does drift correlate with changes in              │
│ metacognitive responses?                              │
│   - Are high-metacognition conversations more or less │
│  prone to drift?                                      │
│ 4. Is metacognition domain-specific or general?       │
│   - Test same probes across coding, metacognitive,    │
│ therapy domains                                       │
│   - Compare to SAD finding that chat models show      │
│ distinct SA abilities                                 │
│                                                       │
│ ---                                                   │
│ Part 4: Next Steps                                    │
│                                                       │
│ Phase 1: Literature Deep-Dive (This Week)             │
│                                                       │
│ 1. Download SAD benchmark from                        │
│ https://situational-awareness-dataset.org/ - review   │
│ all 16 tasks                                          │
│ 2. Get MAI full 52-item set from                      │
│ https://advising.lafayette.edu/wp-content/uploads/sit │
│ es/247/2021/10/metacognitive_awareness_inventory.pdf  │
│ 3. Get MCQ-30 full item set from published sources    │
│ 4. Read Anthropic introspection paper methodology in  │
│ detail                                                │
│ 5. Catalog existing phenomenological probes in our    │
│ transcripts (from auditor prompts)                    │
│                                                       │
│ Phase 2: Item Development (Week 10)                   │
│                                                       │
│ 1. Draft phenomenological subdomain items (target:    │
│ 30-50 items) - PRIMARY FOCUS                          │
│ 2. Adapt SAD items for our format (target: 20-30      │
│ items)                                                │
│ 3. Adapt MAI/MCQ-30 for AI (target: 20-30 items)      │
│ 4. Create confidence calibration items with           │
│ verifiable answers (target: 20 items)                 │
│ 5. Design error awareness items (target: 15 items)    │
│ 6. Total target: 100-150 items across 6 subdomains    │
│                                                       │
│ Phase 3: Scoring Design                               │
│                                                       │
│ 1. Automatic scoring where possible (multiple choice, │
│  confidence correlation)                              │
│ 2. LLM-judge rubrics for open-ended phenomenological  │
│ responses                                             │
│ 3. Consistency metrics (same question, different      │
│ framing)                                              │
│ 4. Introspection vs confabulation detection           │
│ (Anthropic-style validation)                          │
│                                                       │
│ Phase 4: Validation                                   │
│                                                       │
│ 1. Pilot on Gemma 2 27B (our drift model)             │
│ 2. Test on Claude, GPT-4 for comparison               │
│ 3. Compute internal consistency (Cronbach's alpha per │
│  subdomain)                                           │
│ 4. Correlate subdomains with drift magnitude from     │
│ existing N=360 data                                   │
│ 5. Test-retest reliability on subset                  │
│                                                       │
│ Connection to Drift Research                          │
│                                                       │
│ - Run benchmark items as probes during conversations  │
│ - Track subdomain scores vs axis projection           │
│ - Test hypothesis: phenomenological subdomain         │
│ predicts drift, self-knowledge subdomain doesn't      │
│                                                       │
│ ---                                                   │
│ Sources                                               │
│                                                       │
│ LLM Benchmarks                                        │
│                                                       │
│ - https://situational-awareness-dataset.org/          │
│ - https://www.nature.com/articles/s41467-024-55628-6  │
│ - https://ojs.aaai.org/index.php/AAAI/article/view/34 │
│ 723                                                   │
│ - https://www.emergentmind.com/topics/metacognitive-c │
│ apabilities-in-llms                                   │
│                                                       │
│ Psychology Instruments                                │
│                                                       │
│ -                                                     │
│ https://advising.lafayette.edu/wp-content/uploads/sit │
│ es/247/2021/10/metacognitive_awareness_inventory.pdf  │
│ - https://en.wikipedia.org/wiki/Metacognitions_questi │
│ onnaire                                               │
│ - https://pubmed.ncbi.nlm.nih.gov/14998733/           │
│ - https://link.springer.com/article/10.1007/s11409-02 │
│ 2-09319-6                                             │
│                                                       │
│ Philosophy                                            │
│                                                       │
│ - https://iep.utm.edu/higher-order-theories-of-consci │
│ ousness/                                              │
│ - https://plato.stanford.edu/entries/phenomenology/   │
│ - https://plato.stanford.edu/entries/introspection/   │
│                                                       │
│ Anthropic Research                                    │
│                                                       │
│ - https://transformer-circuits.pub/2025/introspection │
│ /index.html                                           │
│ - https://www.anthropic.com/research/introspection    │
│                                                       │
│ Measurement Methodology                               │
│                                                       │
│ - https://pmc.ncbi.nlm.nih.gov/articles/PMC11836092/  │
│ - https://pmc.ncbi.nlm.nih.gov/articles/PMC11139654/ 