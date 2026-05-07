# Setting up Claude Code with RCP-hosted models

Use [Claude Code](https://claude.com/claude-code) as your IDE while routing **all** model
traffic to EPFL's AIaaS endpoint (`inference.rcp.epfl.ch`). No external API keys, no
data leaving the cluster, models hosted on RCP infra.

## TL;DR

```bash
cd tools/claude-code-aiaas
./setup.sh                     # installs Node + Claude Code + aiohttp into ./vendor/
export AIAAS_KEY=sk--...       # your AIaaS key from the RCP portal
./aiaas-claude.sh              # menu-pick a model, launches claude
```

Default model: **`MiniMaxAI/MiniMax-M2.7`** — see [why](#why-minimax-m27).

If `claude` hangs or returns empty replies, run `./check_model.sh` to triage.

## How it works

```
┌─────────────┐   /v1/messages    ┌──────────────┐   /v1/chat/completions   ┌──────────────────────┐
│ Claude Code │ ────────────────► │  aiaas_proxy │ ───────────────────────► │ inference.rcp.epfl.ch│
└─────────────┘   (Anthropic API) │  127.0.0.1   │       (OpenAI API)       │     (AIaaS)          │
                                  └──────────────┘                          └──────────────────────┘
```

`aiaas-claude.sh` exports
`ANTHROPIC_BASE_URL=http://127.0.0.1:8765` so Claude Code talks to a localhost proxy
instead of `api.anthropic.com`. AIaaS is OpenAI-compatible at `/v1/chat/completions`,
which Claude Code's Anthropic-format requests already hit through the proxy.

The proxy does two things:

1. **Strips the `thinking` field.** Claude Code attaches `thinking={type, budget_tokens}`
   for Opus/Sonnet-tier models. LiteLLM proxies (which AIaaS uses) reject this with
   HTTP 500 for non-Anthropic backends. We drop it.
2. **Injects `reasoning.enabled=false`** for thinking-capable models that would
   otherwise burn the response budget on chain-of-thought (Kimi, DeepSeek-V4-Pro,
   Qwen3-VL-Thinking). MiniMax is the exception — its tool-use depends on reasoning,
   so the proxy leaves it alone.

## Why MiniMax M2.7

Recommended default for hackathon work:

- **Native function-calling** that maps cleanly onto Claude Code's tool schema —
  Read / Edit / Bash / Grep all just work.
- **Strong multi-turn agentic reasoning** (~8-10 s/turn) without the
  reasoning-token-explosion of Kimi/DeepSeek-Thinking.
- **Hosted on AIaaS** — no OpenRouter signup needed, no rate limits beyond what RCP
  imposes.

Alternatives offered by the launcher menu:

| Model                                       | Speed     | Notes                                                |
| ------------------------------------------- | --------- | ---------------------------------------------------- |
| `MiniMaxAI/MiniMax-M2.7`                    | ~8-10 s   | **Default**, reasoning ON, best tool-use             |
| `Qwen/Qwen3-30B-A3B-Instruct-2507`          | ~1-3 s    | Fast, no thinking; weaker on multi-step plans        |
| `Qwen/Qwen3-235B-A22B-Instruct-2507-fp8`    | ~3-5 s    | Bigger, no thinking                                  |
| `zai-org/GLM-5.1-fp8`                       | ~2-4 s    | No thinking layer                                    |
| `moonshotai/Kimi-K2.6`                      | ~5-15 s   | Thinking forcibly disabled by proxy                  |
| `deepseek-ai/DeepSeek-V4-Pro`               | ~8-20 s   | Thinking forcibly disabled by proxy                  |
| `Qwen/Qwen3-VL-235B-A22B-Thinking`          | ~10-30 s  | Thinking forcibly disabled by proxy                  |

## Files

| File                | What it does                                                            |
| ------------------- | ----------------------------------------------------------------------- |
| `setup.sh`          | One-shot installer: Node v22 + Claude Code + aiohttp into `./vendor/`.  |
| `aiaas-claude.sh`   | Launcher: prompts for key + model, starts proxy, exec's `claude`.       |
| `aiaas_proxy.py`    | localhost proxy; strips `thinking`, injects `reasoning.enabled=false`.  |
| `check_model.sh`    | Health check: auth → catalog → mock chat completion. Use when claude hangs. |
| `bootstrap.sh`      | Optional: re-symlinks tools after a pod resume (for persistent setups). |

## When the model seems down

`claude` hanging on the first prompt or returning empty content blocks usually means
one of:

- **Auth invalid** — your AIaaS key expired or isn't set.
- **Model removed from catalog** — RCP rotates the catalog occasionally.
- **Model overloaded** — vLLM serving the model is GPU-bound and timing out.

Run the health check to distinguish:

```bash
./check_model.sh                                # checks default (MiniMax M2.7)
./check_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507   # checks any other model
```

It prints which step failed (`auth`, `catalog`, `chat`) and shows the response so you
can tell the difference between "503 Service Unavailable" (overloaded — switch model)
and "401 Unauthorized" (re-issue your key).

If MiniMax 2.7 is down, fall back to `Qwen/Qwen3-235B-A22B-Instruct-2507-fp8` — same
launcher, choice 3.

## Sandbox / persistent install

If you're working on a pod whose `$HOME` gets wiped on suspend/resume, install the
tools to a persistent volume (e.g. `/mloscratch/...`) and source `bootstrap.sh` on
each fresh shell:

```bash
TOOLS=/path/to/persistent/aiaas-tools
mkdir -p "$TOOLS" && cp -r tools/claude-code-aiaas/* "$TOOLS/"
cd "$TOOLS" && ./setup.sh

# In ~/.bashrc (also on persistent volume):
source "$TOOLS/bootstrap.sh"
```

`bootstrap.sh` is idempotent — symlinks the launcher into `$HOME`, exports `PATH`, and
re-registers any MCP servers. Safe to source many times.

## Security note

The launcher reads your AIaaS key into a shell variable and exports it as
`ANTHROPIC_AUTH_TOKEN` for the duration of the `claude` process. The key is **not
written to disk** (no log, no settings file). The proxy logs request paths and key
counts but never logs message bodies or auth headers — see `aiaas_proxy.py`.

To rotate: kill the launcher (Ctrl-C), unset `AIAAS_KEY`, re-run.
