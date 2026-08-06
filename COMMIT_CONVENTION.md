# Commit Convention

FinSight follows [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <subject>

[optional body]
```

## Types

| Type       | Use for |
|------------|---------|
| `feat`     | New capability (a node, an agent, an endpoint) |
| `fix`      | Bug fix |
| `refactor` | Restructuring with no behaviour change |
| `graph`    | LangGraph topology, state schema, or reducer changes |
| `prompt`   | Prompt-constant edits in a `config.py` |
| `data`     | Data-source wrappers, caching, rate limits |
| `vec`      | Qdrant collections, chunking, embeddings, ingest |
| `eval`     | Eval datasets, evaluators, threshold sweeps |
| `test`     | Tests only |
| `docs`     | Documentation only |
| `perf`     | Performance work |
| `config`   | Tooling, lint, type-check configuration |
| `ci`       | CI/CD pipelines |
| `infra`    | Docker, compose, deployment |
| `chore`    | Housekeeping |

## Scopes

`core`, `data`, `vec`, `research`, `monitor`, `dedup`, `persistence`, `api`, `ui`, `eval`, `infra`

## Examples

```
feat(research): add Send-based dynamic fan-out over agent x ticker
graph(monitor): add human_approval interrupt node before dispatcher
vec(dedup): strip volatile numerics from canonical_text before embedding
prompt(research): mandate inline [SRC:...] markers in synthesizer
eval(dedup): sweep TAU_HIGH and pick at precision >= 0.97
fix(data): send SEC User-Agent on every request, not just submissions
infra(vec): pin qdrant to v1.17.1 on :6335
```

## Phase gate rule

Do not start phase N+1 until phase N is **demoable, tested, and committed**.
Tag each phase boundary:

```
git tag -a phase-0 -m "Setup gate: skeleton, isolated Qdrant, traced Gemini call"
```
