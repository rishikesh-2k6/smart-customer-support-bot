import json
import time
from groq import Groq
from config import (
    GROQ_API_KEYS, MODEL, MAX_TOKENS, TEMPERATURE,
    TOP_K, SIMILARITY_THRESHOLD, VALID_STATUSES, VALID_REQUEST_TYPES, COMPANIES
)
from classifier import classify
from retriever import search
from prompts import SYSTEM_PROMPT, build_user_prompt, build_no_match_prompt
from logger import log_turn

# --- Key rotation pool ---
_clients = []
_current_key_idx = 0

def _init_clients():
    global _clients
    if not _clients:
        _clients = [Groq(api_key=key) for key in GROQ_API_KEYS]
        print(f"[agent] Loaded {len(_clients)} API keys for rotation")

def _get_client():
    global _current_key_idx
    _init_clients()
    return _clients[_current_key_idx % len(_clients)]

def _rotate_key():
    global _current_key_idx
    _current_key_idx += 1
    idx = _current_key_idx % len(_clients)
    print(f"\n[agent] Rotated to API key #{idx + 1}/{len(_clients)}")
    return _clients[idx]


def _infer_company(issue, subject, declared_company):
    if declared_company and declared_company.lower() not in ("none", ""):
        return declared_company.lower()

    combined = (issue + " " + (subject or "")).lower()

    hackerrank_keywords = [
        "hackerrank", "hacker rank", "screen", "interview", "candidate", "assessment",
        "test", "coding challenge", "skill", "hire", "skillup", "chakra", "proctoring",
        "leaderboard", "question library", "plagiarism"
    ]
    claude_keywords = [
        "claude", "anthropic", "workspace", "claude.ai", "opus", "sonnet", "haiku",
        "api key", "prompt", "artifact", "team plan", "pro plan", "conversation",
        "context window", "constitutional ai", "mcp", "claude code"
    ]
    visa_keywords = [
        "visa", "card", "payment", "transaction", "dispute", "chargeback", "atm",
        "pin", "blocked card", "traveller cheque", "travel money", "contactless",
        "international", "foreign currency", "bank", "debit", "credit"
    ]

    scores = {"hackerrank": 0, "claude": 0, "visa": 0}
    for kw in hackerrank_keywords:
        if kw in combined:
            scores["hackerrank"] += 1
    for kw in claude_keywords:
        if kw in combined:
            scores["claude"] += 1
    for kw in visa_keywords:
        if kw in combined:
            scores["visa"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def _call_llm(system, user, ticket_id):
    _init_clients()
    num_keys = len(_clients)
    max_rotations = 3  # Try 3 full rotations through all keys
    keys_tried_this_rotation = 0

    for attempt in range(num_keys * max_rotations):
        client = _get_client()
        try:
            log_turn(ticket_id, "llm_prompt", {"system_len": len(system), "user_len": len(user), "attempt": attempt})

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()
            log_turn(ticket_id, "llm_response", {"raw": raw[:500]})

            # Strip accidental markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            log_turn(ticket_id, "error", {"type": "json_parse", "error": str(e), "attempt": attempt})
            continue
        except Exception as e:
            error_str = str(e)
            log_turn(ticket_id, "error", {"type": "api_error", "error": error_str[:200], "attempt": attempt})
            if "429" in error_str or "rate" in error_str.lower():
                _rotate_key()
                keys_tried_this_rotation += 1
                # After trying all keys, wait for rate limit reset
                if keys_tried_this_rotation >= num_keys:
                    keys_tried_this_rotation = 0
                    print(f"\n[agent] All keys exhausted, waiting 60s for reset...")
                    time.sleep(60)
                else:
                    time.sleep(1)
                continue
            # Other error
            _rotate_key()

    return None


def _validate_result(result):
    if not isinstance(result, dict):
        return _fallback_result("LLM returned non-dict response.")

    status = result.get("status", "").lower().strip()
    if status not in VALID_STATUSES:
        status = "escalated"

    req_type = result.get("request_type", "").lower().strip()
    if req_type not in VALID_REQUEST_TYPES:
        req_type = "product_issue"

    product_area = str(result.get("product_area", "general")).strip() or "general"
    response = str(result.get("response", "")).strip()
    justification = str(result.get("justification", "")).strip()

    if not response:
        response = "Your request has been received and escalated to our support team."
        status = "escalated"

    if not justification:
        justification = "Automated triage decision based on ticket content and documentation retrieval."

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": req_type
    }


def _fallback_result(reason):
    return {
        "status": "escalated",
        "product_area": "general",
        "response": "We encountered an issue processing your request. A support agent will follow up shortly.",
        "justification": f"Fallback due to processing error: {reason}",
        "request_type": "product_issue"
    }


def triage(row):
    issue = str(row.get("issue", row.get("Issue", ""))).strip()
    subject = str(row.get("subject", row.get("Subject", ""))).strip()
    declared_company = str(row.get("company", row.get("Company", ""))).strip()
    ticket_id = str(row.get("id", row.get("ticket_id", "unknown")))

    log_turn(ticket_id, "input", {
        "issue": issue[:200],
        "subject": subject,
        "company": declared_company
    })

    # --- Step 1: Rule-based pre-screening ---
    classifier_result = classify(issue, subject)
    if classifier_result is not None:
        log_turn(ticket_id, "classifier", {
            "decision": "hard_stop",
            "trigger": classifier_result.get("trigger"),
            "status": classifier_result["status"]
        })
        output = {k: v for k, v in classifier_result.items() if k != "trigger"}
        log_turn(ticket_id, "output", output)
        return output

    log_turn(ticket_id, "classifier", {"decision": "pass", "proceeding_to_rag": True})

    # --- Step 2: Infer company ---
    company = _infer_company(issue, subject, declared_company)
    log_turn(ticket_id, "company_inference", {"declared": declared_company, "inferred": company})

    # --- Step 3: Retrieve relevant docs ---
    chunks = search(query=issue, company=company, top_k=TOP_K)
    log_turn(ticket_id, "retriever", {
        "company_filter": company,
        "num_results": len(chunks),
        "top_score": chunks[0]["score"] if chunks else 0,
        "top_sources": [c["source"] for c in chunks[:3]]
    })

    # --- Step 4: Check similarity threshold ---
    top_score = chunks[0]["score"] if chunks else 0.0
    if top_score < SIMILARITY_THRESHOLD:
        user_prompt = build_no_match_prompt(issue, subject, declared_company)
        chunks_used = []
    else:
        user_prompt = build_user_prompt(issue, subject, declared_company, chunks)
        chunks_used = chunks

    # --- Step 5: Call LLM ---
    raw_result = _call_llm(SYSTEM_PROMPT, user_prompt, ticket_id)

    if raw_result is None:
        result = _fallback_result("LLM API call failed after retry.")
    else:
        result = _validate_result(raw_result)

    # --- Step 6: Log and return ---
    log_turn(ticket_id, "output", result)
    return result
