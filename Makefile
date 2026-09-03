PY ?= python3

# Execute the notebook against the frozen data in data/. No API key needed.
reproduce:
	cd notebooks && $(PY) -m nbconvert --to notebook --execute anyone.ipynb --output anyone.ipynb

# Regenerate every file in data/ from scratch. Needs SUITE_ANTHROPIC_KEY in .env
# and the private SUITE corpus under ext/. About 900 API calls, roughly $3.
rerun:
	$(PY) src/build_items.py
	$(PY) src/generate.py
	$(PY) src/judge_humanlm.py
	$(PY) src/probe.py
	$(PY) src/embed.py

.PHONY: reproduce rerun
