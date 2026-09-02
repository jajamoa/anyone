# How human/user simulation papers validate fidelity (survey, 2026-09-02)

Question: does the fidelity metric depend on WHICH person is simulated? Each row was checked against the arXiv abstract plus full text or journal page unless marked unverified.

Metric codes: A = free-text similarity (embedding / ROUGE / LLM-judge vs the person's actual reply); B = closed-form answer accuracy vs the person's own answer; C = behavioral log-likelihood; D = distributional / population match; E = human evaluation / Turing-style.

| Paper (year, venue) | Task / output | Primary metric | Identity control? | Note |
|---|---|---|---|---|
| HumanLM, Wu et al. 2026, arXiv 2603.03303 | Free text: next reply (Humanual, 26k users) | A: LLM-judge key-point coverage; embedding cosine; state alignment; 111-person similarity study (E) | No. Baselines vary training, not profile | Latent states scored against states the judge infers from the reply text |
| Turing-RL, Wang et al. 2026, arXiv 2606.19336 | Free text: Reddit (ConvoKit) and PRISM chat | E-style LLM Turing judge (Sonnet 4.6) + HumanLM similarity + specificity; 360-person Prolific Turing test | No. Ablation over history / persona / both only; Turing score barely moves. No wrong-user condition | Reddit human Turing test at chance for all trained models; human GT accuracy 0.38 to 0.41 |
| Park et al. 2024/2026, arXiv 2411.10109 | Closed: GSS, Big Five, econ games, 5 replications (1,052 people) | B, normalized by 2-week retest (0.85) | Partial: demographics-only, persona paragraph, interview lesion. No shuffle | Reasoning field discarded: "no ground truth for those" |
| Twin-2K-500, Toubia et al. 2025, arXiv 2505.17479 | Closed: 88 held-out heuristics-and-biases items | B (71.7% vs 81.7% retest) + D | No. Only uniform random | Twins collapse to normative answers |
| Peng et al. 2025 "Funhouse Mirrors", arXiv 2509.19088 | Closed: 164 outcomes, 19 studies, Twin-2K people | B (r = 0.20 per outcome) + D | Yes: empty persona (r = 0.08), demographics-only (r = 0.145), random (0). No cross-person permutation | Full-persona twins closer to empty-persona twins (MAD 0.175) than to own humans (0.252) |
| Centaur, Binz et al. 2025, Nature | Behavioral: trial-level choices, 160 experiments | C: held-out NLL | None; no persona at all | Population-conditional |
| UserLM, Naous et al. 2025, ICLR 2026 | Free text: user turns given intent | D-like: perplexity; 6 behavioral statistics | N/A: no individual target | Personalization is future work |
| OpinionQA, Santurkar et al. 2023, ICML | Closed: Pew MCQ log-probs | D: Wasserstein to group marginals | Partial: conditioned vs not | Never predicts an individual |
| Argyle et al. 2023, Political Analysis | Mixed: free text + vote + ANES chains | D (algorithmic fidelity); Study 1 E; Study 2 de facto B | Yes: no-backstory ablation, demographics-only. No shuffle | Declines to report individual percent-correct |
| Anthology, Moon et al. 2024, EMNLP | Closed: Pew MCQ from backstories | D: Wasserstein, correlation Frobenius | Yes: random backstory matching, 18% / 27% worse | Closest analogue to a cross-person shuffle for prompted personas |
| Hewitt et al. 2024, working paper | Closed: demographic-profile predictions averaged per condition | D: r = 0.85 over 476 effects | Partial, negative: treatment x demographic interactions r = -0.01, 0.16, -0.03 | Group fidelity with almost no identity signal |
| Aher et al. 2023 "Turing Experiments", ICML | Behavioral, synthetic names | D: replication of effect curves | Limited: name consistency | No real person |
| Dominguez-Olmedo et al. 2024, NeurIPS | Closed: 25 ACS items | D + diagnostics; discriminator > 90% | Yes, negative: alignment tracks subgroup entropy | Undercuts OpinionQA metric |
| Bisbee et al. 2024, Political Analysis | Closed numeric: feeling thermometers, 7,530 ANES personas | D; secondary per-respondent MAE | Yes: demographics vs politics vs full. No shuffle | Explanations checked for coherence only |
| Kim & Lee 2023, arXiv 2305.09620 | Closed: binary GSS, fine-tuned with respondent embeddings | B: AUC 0.857 | Yes, closest to cross-person: shuffling respondent embeddings drops AUC to 0.720 | Not a prompted persona simulator |
| Hu & Collier 2024, ACL | Closed: subjective annotation labels | B-ish: variance explained | Partial: no-persona baseline | Personas explain < 10% of variance |
| SubPOP, Suh et al. 2025, ACL | Closed: subpopulation distributions | D | Partial | Group-level by design |
| Chuang et al. 2024, NAACL Findings | Free text tweets, synthetic personas | D: opinion distribution | Partial: no-persona control | No per-person target |
| PersonaGym, Samuel et al. 2025, EMNLP Findings | Free text, 200 synthetic personas | A/E: LLM rubric | No real-person GT | Action Justification judged for plausibility only |
| Character-LLM, Shao et al. 2023, EMNLP | Free text, 9 figures | A/E: GPT-3.5 judge | Partial: hallucination probes | No real utterances |
| RoleLLM, Wang et al. 2024, ACL Findings | Free text, 100 roles | A: Rouge-L vs GPT-4 references | Partial: unseen roles | Fidelity to GPT-4's rendition |
| CharacterEval, Tu et al. 2024, ACL | Free text, 77 characters | E/A + MBTI back-test (B) | Partial | |
| SOTOPIA, Zhou et al. 2024, ICLR | Free text + actions, 40 characters | E/A: GPT-4 judge | No | Social competence, not fidelity |
| PersonaHub, Ge et al. 2024 | Synthetic data | None of A to E | No | |
| USimAgent, Zhang et al. 2024, SIGIR | Behavioral: search sessions | A + B: BLEU vs query, click/stop F1 | No | Reasoning never scored |
| DAUS, Sekulic et al. 2024 | Free text user turns (MultiWOZ) | Goal fulfillment; A secondary | N/A | Users are goals |
| Zhu et al. 2024, WWW Companion | Free text user turns, CRS | Downstream Recall@k | No | |
| OPeRA, Lu et al. 2025, arXiv 2503.20749 | Behavioral: web actions, 51 users | B: next-action accuracy | No | 604 human rationales collected, unused |
| BehaviorChain, Li et al. 2025, ACL Findings | Closed: 4-way next-behavior MCQ, 1,001 personas | B | No | Characters, not living users |
| TwinVoice, Du et al. 2025, arXiv 2510.25536 | Both: 4-way pick-the-real-reply + generation | B (76%); A/E secondary | No | Reasons not graded vs the person's |
| PersonaBench, Tan et al. 2025, ACL Findings | Short QA from private-data RAG | B-ish | No | Synthetic people |
| Lost in Simulation, Seshadri et al. 2026, arXiv 2601.17087 | Simulated vs 451 humans on tau-Bench | Calibration of success rates | No | |
| Kinzinger & Hartmann 2026 (SOEP); Jia et al. 2026 (LISS) | Closed: held-out panel items | B individual (78.8%, r = 0.59) + D | Not found in abstracts (unverified) | Identity signal concentrated in low-variability items |

## Synthesis

The field splits by output type. Closed-answer simulators validate with individual accuracy normalized by test-retest; the survey and social-science line validates distributionally and rarely scores individuals; free-text simulators with a real reply (HumanLM, Turing-RL, TwinVoice generation, USimAgent, RoleLLM) use similarity or an LLM judge. Similarity is therefore not the field default, only the default for free-text simulators with a real-person target, which is where HumanLM and Turing-RL sit.

No paper in this set evaluates reasoning structure against a justification the person actually gave. HumanLM's latent states are inferred by the judge from the reply text. PersonaGym, TwinVoice, OPeRA and Bisbee score generated reasons for plausibility only. Park discards the reasoning field for lack of ground truth.

Identity controls are rare and one-sided. The common pattern is a nested-information ablation (demographics-only, empty persona, interview lesion), which shows that more data about the right person helps but never asks whether data about the wrong person would score as well. The only respondent-level permutation found is Kim & Lee 2023 (AUC 0.857 to 0.720); Anthology's random-backstory matching is the nearest analogue for prompted personas. Peng et al. (twins closer to the empty-persona twin than to their own human) and Hewitt et al. (interaction effects r near 0) are the strongest published hints that a good fidelity number can carry almost no identity information. HumanLM, Turing-RL, TwinVoice, BehaviorChain, Twin-2K-500 and OPeRA report no identity baseline at all. A cross-person permutation applied to a free-text fidelity metric is an open gap.

## Related claims: judges and similarity metrics miss individuality (searched 2026-09-02)

Nobody states the claim in our form (LLM-judge and embedding similarity reward the shared part and are blind to who wrote it). Four papers from 2025 to 2026 each cover a piece. The specific control, swap the person and rescore free text, appears to be open.

Closest:
1. Guo et al. Individual Turing Test. SIGIR 2026, arXiv:2603.01289. One volunteer, 10+ years of private messages. Strangers prefer simulated replies (~40% vs ~20% for ground truth); acquaintances reverse it. "General human-likeness does not imply identity-specific fidelity." Same claim in spirit; metrics not tested directly.
2. Wang et al. Learning User Simulators with Turing Rewards (Turing-RL). arXiv:2606.19336. "Content matching and human-likeness come apart": similarity reward raises ground-truth coverage without making replies harder to distinguish from the real user. Closest in mechanism; framed as single-reference variance.
3. Jangra et al. Evaluating Style-Personalized Text Generation. EMNLP 2026, arXiv:2508.06374. BLEU, ROUGE, style embeddings and GPT-4.1 judge all degrade from domain to author to personalized-vs-generic (16% then 7.5%). Same claim on the metric side; writing assistance, not simulation.
4. Xiao et al. The Chameleon's Limit. arXiv:2604.24698. "Fidelity Trap": highest per-persona fidelity (rho > 0.9) coincides with most collapsed populations (d > 6). Per-persona scores reward homogenization; trait scores, not free text.

Partial:
5. Zhou et al. PersonaEval. COLM 2025, arXiv:2508.10014. LLM judges identify the speaking character at ~69% vs humans 90.8%. Fictional characters.
6. Bao et al. Eval4Sim. arXiv:2603.02876. BLEU/perplexity capture surface overlap; measures consistency via authorship verification across personas.
7. Li et al. PRISM. EMNLP 2026, arXiv:2608.26674. Holistic LLM-judge persona scoring as "appraisal hallucination"; structured inverse inference instead.
8. Shin et al. Spotting Out-of-Character Behavior. Findings ACL 2025, arXiv:2506.19352. Whole-response scores hide persona drift; atomic-level scoring.

Tangential:
9. Groner and Chiou. arXiv:2606.16778. GPT-4o as writer and judge fails to reproduce individual style preference variation (n = 30).
10. Abbas. Attribution Quality in AI-Generated Content. arXiv:2510.13898. GPT-4o attribution judge 68% vs style embeddings 82%; human-vs-machine, not person-vs-person.
11. Huang, Chen, Shu. Can LLMs Identify Authorship? Findings EMNLP 2024, arXiv:2403.08213. Counter-evidence to keep in mind: an LLM explicitly asked for authorship does reasonably well. A similarity judge is never asked.
12. Taday Morocho et al. Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents. WWW 2026 Companion, arXiv:2602.18462. WVS, 70k cases; persona prompting gives no clear gain; cleanest "drop the persona, little changes" control, but closed-form items. See also arXiv:2604.28048.
13. Hu et al. SimBench. arXiv:2510.17516. Alignment-simulation tradeoff: tuned models do well on consensus items, badly where humans disagree. Generator side.
14. Wang, Morgenstern, Dickerson. Nature MI 2025, arXiv:2402.01908. LLM stand-ins flatten within-group variation. Standard homogenization citation.
15. Twin-2K-500 (arXiv:2505.17479), Park et al. 2024 (1,000 people), TwinVoice (arXiv:2510.25536): closed-form or judge-scored, none tests judge sensitivity to identity.
