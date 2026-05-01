import json
import datetime
from config import LOG_PATH, LOG_DIR

def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_turn(ticket_id, stage, data):
    """
    Log a single event to log.txt.
    stage: one of 'classifier', 'retriever', 'llm_prompt', 'llm_response', 'output', 'error'
    data: any dict or string
    """
    _ensure_log_dir()
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ticket_id": ticket_id,
        "stage": stage,
        "data": data
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
