# Anyone Scores the Same

Similarity rewards for user simulators are invariant to the user.

Project page: https://jajamoa.github.io/anyone/

Free-text user simulators (HumanLM, Turing-RL's Sim-RL baseline) are trained and scored by an LLM judge that compares a generated reply to the person's real one. Feed the simulator someone else's history and the score does not move. A structured probe of the person's reasoning (SUITE stance / warrant questions) does move.

## Layout

- `docs/` project page (GitHub Pages)
- `src/` pilot pipeline: item construction, generation, HumanLM judge (paraphrased and verbatim replicas), embedding cosine, structured probe, analysis
- `notes/eval-survey.md` how 30 simulator papers validate fidelity, plus related claims
- `results/demo/` judge outputs for the worked example on the page

`src/humanlm_judge_exact.py` reproduces HumanLM's evaluation judge verbatim: prompt from `zou-group/humanlm` (`humanlm/metrics/state_reward.py`), claude-haiku-4-5, temperature 0, one generation per call.

## Data

The pilot items (real r/AmItheAsshole comments and commenter histories, pseudonymized) are not in this repo. They derive from the SUITE corpus; ask for access.

Scripts expect `SUITE_ANTHROPIC_KEY=...` in a local `.env`.

## Citation

```bibtex
@misc{li2026anyone,
  title  = {Anyone Scores the Same: Similarity Rewards for User Simulators Are Invariant to the User},
  author = {Li, Chance Jiajie},
  year   = {2026},
  note   = {Work in progress}
}
```
