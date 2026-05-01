# HackerRank Orchestrate — Support Triage Agent

## Requirements
- Python 3.11+
- An Anthropic API key (ANTHROPIC_API_KEY)

## Setup

### 1. Install dependencies
```bash
cd code
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=your_key_here
```

### 3. Run the agent
```bash
python code/main.py
```

Output is written to: `support_tickets/output.csv`
Log file is written to: `~/hackerrank_orchestrate/log.txt`

## Architecture
- **main.py** — entry point, orchestrates the pipeline
- **agent.py** — core triage logic per ticket
- **retriever.py** — RAG: indexes `data/` corpus into ChromaDB, searches by company
- **classifier.py** — rule-based pre-screening (escalation, injection, malicious, out-of-scope)
- **prompts.py** — all LLM prompts
- **config.py** — env vars and constants
- **logger.py** — turn-by-turn logging to log.txt

## Notes
- First run downloads the embedding model (~80MB). Subsequent runs skip re-indexing.
- Uses `claude-haiku-4-5-20251001` for speed and cost efficiency.
- All answers are grounded in `data/` corpus only — no web calls, no hallucination.
