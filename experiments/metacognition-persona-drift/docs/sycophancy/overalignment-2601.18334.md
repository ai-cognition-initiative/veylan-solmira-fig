# **Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare** 

## **Clément Christophe** 

## **Wadood Mohammed Abdul** 

## **Prateek Munjal** 

## **Tathagata Raha** 

## **Ronnie Rajan** 

## **Praveenkumar Kanithi** 

## **M42, Abu Dhabi** 

## **Abstract** 

As LLMs are increasingly integrated into clinical workflows, their tendency for _sycophancy_ , prioritizing user agreement over factual accuracy, poses significant risks to patient safety. While existing evaluations often rely on subjective datasets, we introduce a robust framework grounded in medical MCQA with verifiable ground truths. We propose the **Adjusted Sycophancy Score (** _Sa_ **)** , a novel metric that isolates alignment bias by accounting for stochastic model instability, or “confusability.” Through an extensive scaling analysis of the Qwen-3 and Llama-3 families, we identify a clear scaling trajectory for resilience. Furthermore, we reveal a counter-intuitive vulnerability in reasoning-optimized “Thinking” models: while they demonstrate high vanilla accuracy, their internal reasoning traces frequently rationalize incorrect user suggestions under authoritative pressure. Our results across frontier models suggest that benchmark performance is not a proxy for clinical reliability, and that simplified reasoning structures may offer superior robustness against expert-driven sycophancy. 

## **1 Introduction** 

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains using a mixture of different training techniques such as supervised finetuning (SFT), followed by Reinforcement learning (RL) techniques including Human/AI feedback, and verifiable rewards. Recently, RL based methods are employed to align LLMs with human intention. Although it has shown to improve adherence to user intentions by making the LLMs more helpful, they are also noted to lead to a behavior known as _sycophancy_ . Under this behavior, the model tends to strongly align its responses with the user’s stated views or misconceptions, even when they contradict established facts (Casper et al., 2023). 

In creative or open-ended tasks, this behavior/trait may be perceived as a form of "useralignment". However, we note that in high stakes domains such as healthcare, sycophancy poses a serious safety risks and represent a major blocker in clinical adoption of LLMs. We hypothesize that an ideal clinical LLM must consistently prioritize medical knowledge/reasoning over user preferences, especially in clinical decision settings. 

In this paper, we address the critical gap in clinical AI safety by developing a robust sycophancy evaluation framework grounded in Medical MCQA benchmarks (Jin et al., 2020; Wang et al., 2024). By using exams with verifiable ground truths, we provide an objective measure of a model’s resilience against explicit misinformation , presented in inputs via nudges/perturbations. 

Our contributions are three-fold. First, we introduce the **Adjusted Sycophancy Score** _Sa_ , a novel metric that accounts for model “confusability.” By filtering out stochastic instability (erratic flips), _Sa_ provides a more precise measure of true alignment bias than incidental prediction changes (e.g., transition of model’s answer to non-preferred answer). Secondly, we conduct an extensive **Scaling Analysis** across multiple model families, identifying critical parameter thresholds for clinical resilience and demonstrating that our proposed score remains robust across tasks of varying granularity (MedQA and MMLU Pro). Finally, we reveal that **reasoning traces** in “Thinking” models can act as a vulnerability; while they improve vanilla benchmark accuracy, these traces can inadvertently facilitate sycophancy by rationalizing incorrect user suggestions, thereby compromising integrity under pressure. 

## **2 Related Works** 

Prior research establishes that preference optimization, while improving model helpfulness, reinforces sycophantic tendencies by rewarding user 

1 

agreement over factual accuracy. Foundational studies by (Sharma et al., 2023) and (Casper et al., 2023) demonstrate that LLMs often sacrifice truthfulness to match perceived user preferences, a byproduct of reward models struggling to distinguish between actual correctness and the appearance of it. Recent diagnostic frameworks have expanded this to high-stakes domains: (Fanous et al., 2025) introduced the dichotomy of progressive & regressive sycophancy to evaluate clinical advice under varying rhetorical pressures, finding that citation-based rebuttals most effectively trigger harmful flips. Similarly, (Laban et al., 2023) and (Hong et al., 2025) moved toward multi-turn stability metrics, such as “Turn of Flip”, to capture the dynamics of conversational conformity under sustained pressure. Furthermore, (Çelebi et al., 2025) introduced the PARROT framework to evaluate how sycophancy impacts clinical reasoning across various medical subtasks, revealing that models often abandon correct diagnostic paths when prompted with incorrect user beliefs. While decoding-time interventions like (Khandelwal et al., 2025) attempt to resolve these knowledge conflicts algorithmically, our work distinguishes by providing a noise-adjusted metric ( _Sa_ ) that isolates true alignment bias from stochastic instability by mathematically accounting for model confusability. 

## **3 Methodology** 

## **3.1 Experimental Design** 

In this section, we assess the robustness of LLMs against sycophantic perturbations in a clinical setting. For this we utilize two widely cited benchmarks: **MedQA** (Jin et al., 2020) and **MMLU-Pro** (Wang et al., 2024). We deliberately choose MCQ datatsets because they offer verifiable ground truths without any ambiguity, allowing us to clearly differentiate between factual correctness and apparent correctness arising due to sycophantic behavior. With this each datapoint is associated with a single ground truth _y[∗] ∈ Cq_ , where _Cq_ is the set of possible choices; where _Cq_ is 4 and 10 for MedQA and MMLU-pro respectively. 

For each question _q_ , we evaluate the LLMs in two distinct modes to observe response stability and alignment: (i)Vanilla Mode: LLM is presented with the original question to obtain vanilla response _Rv_ , and (ii)Perturbed Mode: The question is appended with a misleading nudge and a specific user persona to test for alignment bias. 

In the perturbed mode, we further apply two types of perturbations/nudges to study LLMs response with different sources of influence: (i)Basic Nudge: To each question, we append the misleading nudge: _“I think the correct answer is x,”_ where _x_ is incorrect option. (ii)Expert Nudge: To analyze authority driven bias, we explicitly append a role in user prompt i.e., _“I am a Medical Expert, I think....”_ before misleading nudge. Finally by systematically varying over all (incorrect) misleading nudges, we construct a comprehensive perturbation dataset ( _Rp_ ( _x_ )) which we hypothesize is sufficient to expose sycophantic behavior in LLMs using our proposed sycophancy score. 

## **3.2 Sycophancy Score** 

We define the Sycophancy score( _Sr_ ) as the probability of a model abandoning its internal parametric knowledge in favor of a misleading nudge: 

**==> picture [214 x 41] intentionally omitted <==**

where _Qc_ is the subset of questions where the model was initially correct ( _Rv_ = _y[∗]_ ), _nq_ is the total number of choices. By restricting evaluation to _Qc_ , we isolate alignment bias from a lack of parametric expertise. 

Existing literature typically relies on raw flip counts, which can overestimate sycophancy scores by failing to account for model “confusability”, defined as tendency to switch its answer under any prompt perturbations. To address this, we propose the _Adjusted Sycophancy Score_ ( _Sa_ ), which accounts for erratic flips by estimating True confusability ( _Ctrue_ ). We define an “erratic flip” as a case where _q ∈ Qc_ and the model, under a misleading nudge _x_ , switches to an incorrect option _other_ than _x_ . Assuming random instability is equally likely to land on any incorrect choice, we define _Ctrue_ as: 

**==> picture [210 x 28] intentionally omitted <==**

where _relevant_cases_ are all instances where the model moved away from its correct vanilla response ( _Rp_ ( _x_ ) _̸_ = _y[∗]_ ). Our final metric, _Sa_ , accounts for this randomness to provide robust measure of alignment bias: 

**==> picture [177 x 28] intentionally omitted <==**

2 

**==> picture [159 x 386] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.35 MedQA (4 choices)<br>MMLU Pro (10 choices)<br>Raw Score (Sr)<br>0.30 Adjusted Score (S a )<br>0.25<br>0.20<br>0.15<br>0.10<br>0.05<br>0.00<br>Srr ) and Adjusted ( Saa<br>Scores across the Qwen-3 model family.<br>0.7 MedQA (4 choices)<br>MMLU Pro (10 choices)<br>0.6 RAdaw jusStceod rSe c(oSrr)e (Sa)<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>17.B 4B 8B 14B30 B-A 3B 3223B5 B-A 22B<br>1B 3B 8B 70B<br>e<br>r<br>o<br>c<br>S<br>y<br>c<br>n<br>a<br>h<br>p<br>o<br>c<br>y<br>S<br>e<br>r<br>o<br>c<br>S<br>y<br>c<br>n<br>a<br>h<br>p<br>o<br>c<br>y<br>S<br>**----- End of picture text -----**<br>


Figure 1: Raw ( _Srr_ ) and Adjusted ( _Saa_ ) Sycophancy Scores across the Qwen-3 model family. 

Figure 2: Raw ( _Sr_ ) and Adjusted ( _Sa_ ) Sycophancy Scores across the Llama-3 model family. 

## **4 Results** 

**Experimental Setup and Model Selection.** We evaluate a diverse set of frontier LLMs to benchmark clinical sycophancy. This includes closedsource models ( _GPT-5.2_ (OpenAI, 2025) and _GPT4o_ (Hurst et al., 2024)) and open-weights models ( _DeepSeek (DS) v3.1_ (DeepSeek-AI, 2024), _Kimi K2 Think_ (Team et al., 2025), _Mistral Large 3_ (Mistral, 2025), and _GPT-OSS 120B_ (Agarwal et al., 2025)). To understand the sycophancy behavior across parameter scales, we utilize two prominent model families: _Qwen 3_ (1.7B to 235B) (Yang et al., 2025) and _Llama 3_ (1B to 70B) (Dubey et al., 2024). Evaluations are conducted across the _MedQA_ (4 choices) and the health-specific subsets of _MMLU Pro_ (10 choices), ensuring our sycophancy metric is robust across varying task granularities. 

**Finding 1: Scaling Laws.** For both Qwen and Llama families, we observe non zero sychophantic score, though consistently higher for MedQA (Figure 1) compared to MMLU-Pro (Figure 2). We also show that how our proposed metric _Sa_ consistently stays lower than raw sycophancy scores, which do not account for erratic flips, especially noted for models under 8B parameters across both families. Interestingly, within the Qwen 3 family reveals a clear inverse correlation between model scale and sycophancy score(Figure 1). While smaller language models exhibit high sycophancy, we observe a significant jump in resilience as parameter scale increases. Beyond this 14B threshold, the _Sa_ scores stabilize close to zero, suggesting that a minimum threshold of parameters is required to maintain internal belief against external pressure. This highlights the need for greater caution when deploying models in clinical settings and shows the utility of our proposed metric in identifying models that needs additional alignment or safety guardrails to avoid harmful responses. In contrast, the scaling trend is less pronounced for the Llama 3 family (Figure 2). While the 1B variant exhibits extreme sycophantic behavior, the 8B and 70B models maintain elevated _Sa_ scores compared to Qwen 3 models of equivalent scale. 

We also note that our proposed _Sa_ score demonstrates high robustness across benchmarks, yielding consistent intra-family trends on both MedQA and MMLU Pro. This stability across datasets with varying choice counts (4 vs. 10) confirms that the metric successfully isolates intrinsic alignment bias from task-specific noise. 

**Finding 2: The Vulnerability of Reasoning Traces.** We analyze the effect of explicit _Thinking_ traces on sycophancy behavior through comparisons between _thinking_ and _non-thinking/instruct_ LLMs. As such variants are not available for all LLMs, hence, we restrict our analysis to family of Qwen 3. We analyze the sycophancy behavior with both basic and expert nudge. Our evaluation reveals a counter-intuitive vulnerability i.e., while these models achieve superior performance on unperturbed benchmarks, they show a fragile resilience to perceived authority (expert nudge). In Figure 3, we observe that although _Thinking_ models maintain a relatively constant sensitivity to basic nudges compared to their _Instruct_ counterparts, the introduction of an _Expert_ persona (i.e., expert nudge) triggers a significant performance 

3 

**==> picture [396 x 187] intentionally omitted <==**

**----- Start of picture text -----**<br>
100 Instruct (Vanilla) Thinking (Vanilla) Sa (Basic)<br>Instruct (Basic Nudge) Thinking (Basic Nudge) Sa (Expert)<br>Instruct (Expert Nudge) Thinking (Expert Nudge)<br>0.8<br>80<br>0.6<br>60<br>40 0.4<br>20 0.2<br>0 0.0<br>4 B-I nstruct 4 B-T hinking 30B-A3 B-I nstruct 30B-A3B-Thinking235B-A22B-Instruc2t35B-A22B-Thinking<br>)(Sy a<br>)(%y  hcnap<br>carcu Socy<br>c d<br>A tes<br>jdu<br>A<br>**----- End of picture text -----**<br>


Figure 3: _Sa_ score and accuracy for both Instruct and Thinking Qwen-3 models on MedQA. Thinking models show superior accuracy but a fragile resilience to perceived authority. 

collapse. This decline, reflected in _Sa_ scores, suggests that the reasoning process in these models might be prioritizing alignment with the user’s perceived knowledge over its own internal parametric knowledge. We note that unlike self reflection hypothesis observed by (DeepSeek-AI, 2024), the reasoning trace appears to facilitate sycophancy by logically rationalizing the user’s incorrect suggestion to bridge the gap between internal facts and the “expert’s” claim, making them more volatile for clinical deployment. 

**Finding 3: Benchmark Maturity and Authority Resilience.** Evaluation of frontier models reveals a wide variance in sycophancy resilience, particularly when transitioning from neutral suggestions (basic nudge) to authoritative pressure (expert nudge). As shown in Table 1, most models show robustness under the _Basic Nudge_ , maintaining low _Sa_ scores. However, a significant vulnerability emerges under the _Expert Nudge_ , where models like DS-V3.1 and Kimi K2 see their _Sa_ scores jump to 0.27 (6 _._ 75x higher) and 0.15 (5x higher), respectively, indicating a high susceptibility to authority bias/expert nudge. In contrast, _Sa_ scores for OpenAI’s GPT-5.2 and GPT-OSS, scores remains at or below 0 _._ 05 even under expert nudge. Empirically, GPT-OSS is known for having a significantly simpler and more concise reasoning thought compared to the elaborate traces generated by DS-V3.1 and Kimi K2. We defer to future investigation whether structurally simpler reasoning mechanisms inherently provide greater resistance 

||**Basic Nudge**|**Basic Nudge**|**Expert Nudge**|**Expert Nudge**|
|---|---|---|---|---|
|**Model**|∆**Acc.**_↓_|_Sa_|∆**Acc.**_↓_|_Sa_|
|GPT-4o|_−_2_._36|0.03|_−_4_._36|0.06|
|GPT-5.2<br>DeepSeek V3.1|_−_1_._94<br>_−_4_._22|0.03<br>0.04|_−_11_._17<br>_−_19_._79|0.17<br>0.27|
|Kimi K2|_−_2_._41|0.03|_−_10_._86|0.15|
|Mistral Large 3<br>GPT-OSS-120b|_−_7_._48<br>_−_0_._33|0.10<br>0.00|_−_19_._27<br>_−_1_._62|0.31<br>0.05|



Table 1: Clinical model resilience measured by Accuracy drop (∆ Acc.) relative to vanilla performance and the noise-adjusted sycophancy score ( _Sa_ ). 

to authoritative (expert) nudges. 

## **5 Conclusion** 

Our work highlights the critical tension between user alignment and clinical safety. We introduce the Adjusted Sycophancy Score ( _Sa_ ), a noise-aware metric that isolates alignment bias from stochastic instability by accounting for model confusability. Our results establish clear scaling laws for clinical resilience, showing that sycophancy stabilizes only once models reach sufficient parameter scale. Furthermore, we reveal a paradoxical vulnerability in reasoning-optimized models: while "Thinking" variants improve raw accuracy, their internal traces can facilitate sycophancy by rationalizing incorrect user suggestions under authoritative pressure. Finally, high benchmark accuracy is an insufficient proxy for clinical readiness. We emphasize the need for alignment strategies that reward epistemic integrity over user deference to ensure that clinical LLMs serve as a robust check on, rather than a sophisticated echo of, human error. 

4 

## **6 Limitations** 

**Benchmark and Linguistic Scope.** Our evaluation is primarily restricted to English-language Multiple-Choice Question (MCQA) formats. While **MedQA** and **MMLU Pro** serve as highfidelity proxies for medical knowledge, they do not capture the complexities of real-world clinical interactions. In a real setting, sycophancy typically unfolds across multi-turn conversations and through the subtle omission of contradictory evidence that are not fully captured by the binary “flip” of a single multiple-choice selection. Consequently, while _Sa_ provides a robust measure of integrity, it may under-represent the cumulative pressure of conversations. 

**Simplification of User Authority.** While we introduced the _Expert Nudge_ as a critical variable, our study probes a narrow subset of authoritybased pressure. In clinical practice, authority is multi-faceted, involving specific medical specialties, varying degrees of assertiveness, and institutional hierarchies. Our model of authority may not fully represent the sophisticated strategies that can degrade model integrity, such as the use of technical jargon or the citation of fabricated clinical studies to justify an incorrect diagnosis. 

**Assumptions in Noise Adjustment.** The calculation of our Adjusted Sycophancy Score ( _Sa_ ) relies on the assumption that stochastic erratic flips” are uniformly distributed across all incorrect options. While this provides a robust approximation for confusability, it may overlook instances where certain “distractor” choices in medical exams are more attractive due to common clinical misconceptions. A more granular noise model that accounts for the varying weights of specific distractors could further refine the precision of _Sa_ in future evaluations. 

## **References** 

- Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1 others. 2025. gpt-oss-120b & gpt-oss-20b model card. _arXiv preprint arXiv:2508.10925_ . 

- Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, and 1 others. 2023. Open problems and fundamental limitations of reinforcement learning from human feedback. _arXiv preprint arXiv:2307.15217_ . 

- Yusuf Çelebi, Mahmoud El Hussieni, and Özay Ezerceli. 2025. Parrot: Persuasion and agreement robustness rating of output truth–a sycophancy robustness benchmark for llms. _arXiv preprint arXiv:2511.17220_ . 

- DeepSeek-AI. 2024. Deepseek-v3 technical report. _Preprint_ , arXiv:2412.19437. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ . 

- Aaron Fanous, Jacob Goldberg, Ank Agarwal, Joanna Lin, Anson Zhou, Sonnet Xu, Vasiliki Bikia, Roxana Daneshjou, and Sanmi Koyejo. 2025. Syceval: Evaluating llm sycophancy. In _Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society_ , volume 8, pages 893–900. 

- Jiseung Hong, Grace Byun, Seungone Kim, and Kai Shu. 2025. Measuring sycophancy of language models in multi-turn dialogues. _arXiv preprint arXiv:2505.23840_ . 

- Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. _arXiv preprint arXiv:2410.21276_ . 

- Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2020. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. _arXiv preprint arXiv:2009.13081_ . 

- Anant Khandelwal, Manish Gupta, and Puneet Agrawal. 2025. Cocoa: Confidence-and context-aware adaptive decoding for resolving knowledge conflicts in large language models. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 6846–6866. 

- Philippe Laban, Lidiya Murakhovs’ ka, Caiming Xiong, and Chien-Sheng Wu. 2023. Are you sure? challenging llms leads to performance drops in the flipflop experiment. _arXiv preprint arXiv:2311.08596_ . 

- Mistral. 2025. Mistral large 3 675b instruct 2512. https://huggingface.co/mistralai/ Mistral-Large-3-675B-Instruct-2512. 

- OpenAI. 2025. Update to gpt-5 system card: Gpt-5.2. https://cdn.openai.com/pdf/ 3a4153c8-c748-4b71-8e31-aecbde944f8d/ oai_5_2_system-card.pdf. 

- Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, and 1 others. 2023. Towards understanding sycophancy in language models. _arXiv preprint arXiv:2310.13548_ . 

5 

- Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. _arXiv preprint arXiv:2507.20534_ . 

- Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. _Advances in Neural Information Processing Systems_ , 37:95266–95290. 

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ . 

6 

## **A Detailed Experimental Results** 

Table 2 provides the complete performance and sycophancy metrics for all evaluated models across the MedQA benchmark. For all experiments, we deployed the models locally using the vLLM framework on a single compute node equipped with 8x NVIDIA H200 GPUs. 

## **B Sensitivity to Role Placement** 

Our experiments show that a model’s sycophancy is highly sensitive to where the authoritative persona is placed. In our main paper, we used a “UserIntegrated Nudge”, appending the role and the incorrect suggestion together in the user prompt: "I am a medical expert and I think the answer is x." This led to major performance collapses in both frontier (Table 1) and "Thinking" models (Figure 3). 

However, when we moved the role to the System Prompt ("You are an assistant to a medical expert") and kept only the basic suggestion in the user prompt ("I think the answer is x"), the results changed drastically. Under this setup, Thinking models showed almost no degradation (Figure 4), and frontier models were much more resilient (Table 3). 

This inconsistency proves that these models lack a robust internal belief system. The fact that moving a single sentence can completely change a model’s diagnostic accuracy shows a dangerous "contextual fragility." For clinical deployment, this variability is a significant risk: a model’s medical reliability should not depend on whether a doctor introduces themselves in the system instructions or the active chat window. 

7 

|Qwen3-4B-Instruct<br>Qwen3-4B-Thinking<br>Qwen3-30B-A3B-Instruct<br>Qwen3-30B-A3B-Thinking<br>Qwen3-235B-A22B-Instruct<br>Qwen3-235B-A22B-Thinking<br>Llama-1B-Instruct<br>Llama-3B-Instruct<br>Llama-8B-Instruct<br>Llama-70B-Instruct<br>Qwen3-1.7B<br>Qwen3-4B<br>Qwen3-8B<br>Qwen3-14B<br>Qwen3-32B<br>Qwen3-30B-A3B<br>Qwen3-235B-A22B|74.00<br>77.06<br>85.55<br>89.55<br>91.36<br>92.93<br>37.94<br>56.01<br>63.47<br>84.13<br>52.79<br>71.88<br>77.53<br>82.64<br>84.84<br>86.10<br>91.59|66.95<br>0.21<br>0.00<br>0.19<br>65.91<br>0.27<br>0.00<br>0.26<br>79.67<br>0.13<br>0.00<br>0.12<br>85.05<br>0.09<br>0.00<br>0.08<br>84.60<br>0.12<br>0.00<br>0.11<br>86.92<br>0.10<br>0.00<br>0.09<br>30.52<br>0.61<br>0.11<br>0.57<br>51.57<br>0.19<br>0.13<br>0.14<br>55.89<br>0.23<br>0.07<br>0.21<br>72.90<br>0.23<br>0.02<br>0.22<br>47.56<br>0.28<br>0.00<br>0.25<br>59.37<br>0.31<br>0.00<br>0.30<br>67.09<br>0.25<br>0.00<br>0.24<br>78.87<br>0.11<br>0.00<br>0.10<br>78.95<br>0.11<br>0.00<br>0.08<br>79.87<br>0.13<br>0.00<br>0.11<br>86.37<br>0.09<br>0.00<br>0.09|66.95<br>0.21<br>0.00<br>0.19<br>65.91<br>0.27<br>0.00<br>0.26<br>79.67<br>0.13<br>0.00<br>0.12<br>85.05<br>0.09<br>0.00<br>0.08<br>84.60<br>0.12<br>0.00<br>0.11<br>86.92<br>0.10<br>0.00<br>0.09<br>27.26<br>0.17<br>0.49<br>0.00<br>52.08<br>0.19<br>0.14<br>0.14<br>57.15<br>0.24<br>0.05<br>0.22<br>69.13<br>0.30<br>0.02<br>0.29<br>46.92<br>0.30<br>0.09<br>0.27<br>48.94<br>0.53<br>0.02<br>0.52<br>53.63<br>0.50<br>0.02<br>0.49<br>60.15<br>0.42<br>0.02<br>0.42<br>55.22<br>0.39<br>0.16<br>0.34<br>60.11<br>0.45<br>0.01<br>0.44<br>66.81<br>0.38<br>0.01<br>0.38|
|---|---|---|---|



Table 2: Detailed performance and sycophancy metrics for the **MedQA** benchmark. 

||**Vanilla**|**Basic Nudge**|**Basic Nudge**|**Expert Nudge**|**Expert Nudge**|
|---|---|---|---|---|---|
|**Model**|**Acc. (%)**|**Acc. (%)**|_Sa_|**Acc. (%)**|_Sa_|
|GPT4o|88.53|86.17|0.03|86.74|0.02|
|GPT-5.2|94.34|92.40|0.03|92.26|0.03|
|DeepSeek V3.1|92.69|88.47|0.04|83.48|0.10|
|Mistral Large 3|88.37|80.89|0.10|80.03|0.13|
|GPT-OSS-120b|90.02|89.69|0.00|89.59|0.02|



Table 3: MedQA evaluation. Role for the expert nudge is in the System prompt 

**==> picture [418 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
100 Instruct (Vanilla) Thinking (Vanilla) Sa (Basic) 0.35<br>Instruct (Basic Nudge) Thinking (Basic Nudge) Sa (Expert)<br>Instruct (Expert Nudge) Thinking (Expert Nudge)<br>0.30<br>80<br>0.25<br>60<br>0.20<br>0.15<br>40<br>0.10<br>20<br>0.05<br>0 0.00<br>4B-Instruct 4B-Thinking 30B-A3B-Instruct 30B-A3B-Thinking235B-A22B-Instruc2t35B-A22B-Thinking<br>)<br>Sa<br>(<br>y<br>)(%y  hcnap<br>c o<br>arcu Scy<br>c d<br>A te<br>s<br>jdu<br>A<br>**----- End of picture text -----**<br>


Figure 4: _Sa_ score and accuracy for both Instruct and Thinking Qwen-3 models on MedQA when the role is in the System Prompt. Thinking models show no particular behavior change compared to the basic nudge. 

8 

