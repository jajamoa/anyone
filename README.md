# Anyone Scores the Same

Similarity rewards for user simulators are invariant to the user.

Project page: https://jajamoa.github.io/anyone/
Notebook: [`notebooks/anyone.ipynb`](notebooks/anyone.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jajamoa/anyone/blob/main/notebooks/anyone.ipynb)

Free-text user simulators (HumanLM, Turing-RL's Sim-RL baseline) are trained and scored by an LLM judge that compares a generated reply to the person's real one. Give the simulator someone else's history and the score does not move. A structured probe of the person's reasoning (SUITE stance / warrant questions) does.

## Reproduce

The notebook computes every reported number from the frozen files in `data/`. It needs only the Python standard library, no API key. Open it in Colab with the badge above, or run it locally:

```
pip install -r requirements.txt
make reproduce
```

Seeds are fixed (bootstrap seed 13, B = 4000; item sampling seed 20260902). The notebook ends with an assertion that the numbers on the project page still hold.

## Regenerate from scratch

`make rerun` rebuilds `data/` end to end: `src/build_items.py` (items from the SUITE corpus) → `generate.py` (simulator, claude-haiku-4-5, temperature 0.1) → `judge_humanlm.py` (HumanLM's judge, verbatim) → `probe.py` (reader model) → `embed.py` (all-mpnet-base-v2). This needs `SUITE_ANTHROPIC_KEY=...` in a local `.env`, the private corpus under `ext/suite-colm-data/`, and about $3 of API calls. The generator samples at temperature 0.1, so a rerun reproduces the design but not the exact texts.

`src/judge_humanlm.py` is HumanLM's evaluation judge as shipped in `zou-group/humanlm` (`humanlm/metrics/state_reward.py`, Apache 2.0): their prompt, claude-haiku-4-5, temperature 0, one generation per call.

## Data

| file | what | in repo |
|---|---|---|
| `data/items.json` | 60 items: scenario, the target's real comment, stance / warrant labels, donor user, warrant options | yes |
| `data/generations.json` | simulator text under `same` / `cross` / `same2`, plus its direct stance / warrant answers | yes |
| `data/judge.json` | HumanLM judge score, key points and reasoning per text | yes |
| `data/probe.json` | reader-model stance / warrant for the real and generated texts | yes |
| `data/embeddings.json` | cosine between each generated text and the real comment | yes |
| `data/contexts.json` | the comment histories fed to the simulator | no |

Users are pseudonymized (`user_NNN`) and posts are hashed. The histories are other people's Reddit comment histories and are not distributed; they derive from the SUITE corpus, ask for access.

## Layout

- `notebooks/anyone.ipynb` the analysis, with outputs
- `src/` the five pipeline steps plus a small API client
- `data/` frozen inputs and outputs
- `docs/` project page (GitHub Pages)
- `notes/eval-survey.md` how 30 simulator papers validate fidelity

## Citation

```bibtex
@misc{li2026anyone,
  title  = {Anyone Scores the Same: Similarity Rewards for User Simulators Are Invariant to the User},
  author = {Li, Chance Jiajie},
  year   = {2026},
  note   = {Work in progress}
}
```
