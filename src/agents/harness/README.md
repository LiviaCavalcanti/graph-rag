# Agent Harness CLI

A self-contained, **offline-first** toolkit to **debug, test, evolve, and evaluate**
the vulnerability-patching agent. No new runtime dependencies; live LLM calls are
opt-in.

```bash
python -m src.agents.harness <command> [options]
```

| Command    | Purpose                                                            | Goal     |
|------------|-------------------------------------------------------------------|----------|
| `variants` | List / diff registered prompt variants                            | evolve   |
| `debug`    | Run **one** scenario and print the full trace, or replay a record | debug    |
| `record`   | Run a suite **live** and save every LLM call to a cassette        | test     |
| `replay`   | Run a suite **fully offline** from a cassette (deterministic)     | test     |
| `eval`     | Score a batch `results.jsonl` or a fixture suite                  | evaluate |
| `compare`  | N-way prompt-variant comparison with deltas                       | evolve   |

---

## Backends (`--backend`)

The backend decides where completions come from. This is what makes every command
runnable with or without the network.

| Value    | Network? | Behaviour |
|----------|----------|-----------|
| `mock`   | No       | Deterministic stub. In `debug` it echoes the scenario's ground-truth patch (→ perfect scores, validates the pipeline). In suite commands it returns a constant stub — use it only as a plumbing smoke test. |
| `replay` | No       | Serves responses from a `--cassette`. Strict: a cache miss is an error. Use for deterministic regression. |
| `record` | Yes      | Calls the live model, then writes each interaction to `--cassette` (replay-if-present). Use to build reusable offline fixtures. |
| `live`   | Yes      | Real Azure/litellm call. Requires `AZURE_API_KEY`, `AZURE_API_BASEURL`, and `MODEL_NAME` in `.env`. |

`replay` and `record` **require** `--cassette PATH`. The `record` and `replay`
commands force their backend, so `--backend` is ignored there.

---

## Common flags

Available on every command except `variants`:

| Flag         | Default                               | Meaning |
|--------------|---------------------------------------|---------|
| `--fixtures` | `tests/fixtures/agent/scenarios.json` | Scenario suite: a JSON file or a directory of `*.json`. |
| `--variant`  | `default`                             | Prompt variant (`default`, `default_v2`, `graph`, `graph_v2`, …). Not on `compare`. |
| `--model`    | `$MODEL_NAME` (else `deepseek-R1`)    | Model/deployment name override. |
| `--backend`  | `mock`                                | `mock` \| `replay` \| `record` \| `live`. |
| `--cassette` | –                                     | Cassette path for `replay`/`record`. |
| `--out`      | –                                     | Write traces/results here (see each command). |

---

## Commands

### `variants` — inspect & diff prompts

```bash
# List every registered variant with its message roles + content fingerprint
python -m src.agents.harness variants

# Unified diff between two variants, message by message
python -m src.agents.harness variants --diff default graph
```

### `debug` — one scenario, full trace

Prints the prompt (all messages), raw model output, parsed patch, and scorecard.

```bash
# Offline: mock echoes the scenario's ground truth → exercises the whole path
python -m src.agents.harness debug --backend mock

# A specific scenario, with the graph-enhanced prompt
python -m src.agents.harness debug --id cwe787-oob-write --variant graph --backend mock

# Against the real model, and also run the LLM judge
python -m src.agents.harness debug --backend live --judge

# Re-score a previously saved InvocationRecord (no LLM call at all)
python -m src.agents.harness debug \
    --from-record path/to/record.json \
    --reference-file path/to/ground_truth.c
```

Extra flags: `--id` (scenario id; default = first), `--judge`, `--from-record`,
`--reference-file`.

### `record` — build an offline cassette (live)

```bash
python -m src.agents.harness record \
    --fixtures tests/fixtures/agent/scenarios.json \
    --variant graph \
    --cassette fixtures/cassettes/graph.jsonl \
    --out runs/record_traces.jsonl
```

Runs the suite against the live model and writes the cassette. Requires Azure keys.

### `replay` — deterministic offline suite

```bash
python -m src.agents.harness replay \
    --fixtures tests/fixtures/agent/scenarios.json \
    --variant graph \
    --cassette fixtures/cassettes/graph.jsonl \
    --out runs/replay_traces.jsonl
```

Prints an aggregate summary (parse rate, exact-match rate, mean similarity/BLEU/
ROUGE/Jaccard, expectation pass rate). With `--out`, writes `*.jsonl` traces plus a
`*.summary.json` sidecar. Add `--judge` to attach LLM verdicts (needs a backend
that can serve the judge call).

### `eval` — score patches

Two modes:

```bash
# Mode A — score an existing batch run (no LLM needed)
python -m src.agents.harness eval --results experiments/output/<run>/results.jsonl --out scores.json

# Mode B — run a fixture suite through a backend, then score it
python -m src.agents.harness eval --fixtures suite.json --backend replay --cassette c.jsonl
```

Mode A reads `generated_patch` / `ground_truth_patch` from each record and prints a
scorecard summary; `--out` writes per-record scorecards. Add `--judge` for the
LLM-as-judge verdict.

### `compare` — A/B (N-way) prompt variants

```bash
# Offline comparison from a cassette, deltas vs the "default" baseline
python -m src.agents.harness compare \
    --variants default,graph,graph_v2 \
    --backend replay --cassette c.jsonl \
    --baseline default \
    --out comparison.json
```

Prints a table of aggregate metrics per variant with `▲`/`▼` deltas against the
baseline, plus the best variant by similarity. Labels are `variant` (or
`variant@model` when `--model` is given), so set `--baseline` to the matching label.

Required: `--variants` (comma-separated). Optional: `--baseline`, `--judge`.

---

## Typical workflows

**Debug a single case offline**

```bash
python -m src.agents.harness debug --id cwe476-null-deref --backend mock
```

**Build once (live), then regress forever (offline)**

```bash
python -m src.agents.harness record --variant graph --cassette c.jsonl   # once, needs Azure
python -m src.agents.harness replay --variant graph --cassette c.jsonl    # in CI, no network
```

**Evolve a prompt and A/B it**

1. Add/register a variant (see [registry.py](registry.py): `register_variant`,
   `register_from_files`, `save_variant_templates`).
2. `record` the new variant to extend the cassette.
3. `compare --variants old,new --backend replay --cassette c.jsonl`.

**Evaluate a real batch run**

```bash
uv run python main.py --mode batch --model gpt-4o --max-queries 20
python -m src.agents.harness eval --results experiments/output/<run>/results.jsonl
```

---

## Fixtures

A suite is JSON — `{"scenarios": [ ... ]}` or a bare list — where each scenario
carries everything `patch_one` needs plus the ground-truth patch and optional
golden `expected` assertions (`min_similarity`, `contains`, `not_contains`,
`exact_match`, `must_parse`, …). See [scenario.py](scenario.py) for the schema and
[../../../tests/fixtures/agent/scenarios.json](../../../tests/fixtures/agent/scenarios.json)
for a worked example.

> All commands are prefixed with `uv run` in this repo, e.g.
> `uv run python -m src.agents.harness variants`.
