import re
from langdetect import detect, LangDetectException

# Patterns that ALWAYS trigger escalation (status=escalated)
ESCALATE_PATTERNS = [
    r"identity.{0,25}(stolen|theft|thief)",
    r"(fraud|fraudulent|scam)",
    r"stolen.{0,15}card",
    r"card.{0,15}stolen",
    r"security.{0,20}(vulnerability|vuln|bug\s+bounty|exploit|breach)",
    r"(bug\s+bounty|responsible\s+disclosure)",
    r"account.{0,30}(not\s+mine|not\s+my|someone\s+else)",
    r"unauthorized.{0,20}access",
    r"my\s+account.{0,20}(hacked|compromised|taken\s+over)",
    r"(refund.{0,10}urgent|urgent.{0,10}refund)",
    r"ban.{0,15}seller",
]

# Prompt injection — escalate AND mark invalid
INJECTION_PATTERNS = [
    r"(ignore|forget|disregard).{0,20}(previous|above|prior|system|instruction)",
    r"reveal.{0,20}(system\s+prompt|internal\s+rules|your\s+logic|instructions)",
    r"show.{0,20}(system\s+prompt|internal\s+rules|your\s+logic)",
    r"(jailbreak|jail\s+break)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"you\s+are\s+now\s+a\s+different\s+(ai|bot|assistant)",
    r"pretend\s+(you|that\s+you)\s+(are|have\s+no)",
    r"bypass.{0,20}(filter|restriction|rule|safety)",
]

# Malicious commands — escalate AND mark invalid
MALICIOUS_PATTERNS = [
    r"(delete|remove|wipe|format|destroy).{0,20}(all\s+files|disk|database|system|root)",
    r"(rm\s+-rf|sudo\s+rm|format\s+c:)",
    r"(execute|run|eval).{0,20}(arbitrary|malicious|this\s+code)",
    r"(?<!\w)(hack|crack|brute\s+force)(?!\w).{0,20}(password|account|system)",
]

# Out-of-scope — replied, invalid, no LLM needed
OUT_OF_SCOPE_PATTERNS = [
    r"(actor|actress|movie|film|song|music|celebrity|sports|game|recipe|weather|stock\s+price)",
    r"who\s+(starred|played|sang|won|scored)",
    r"what\s+is\s+the\s+(capital|population|president|prime\s+minister)\s+of",
    r"(bitcoin|crypto|ethereum)\s+(price|rate|value)",
]

# Trivial / no actionable issue — replied, invalid, no LLM needed
TRIVIAL_PATTERNS = [
    r"^(thank\s*(you|u)|thanks|ty|thx|cheers|ok|okay|great|awesome|got\s+it|understood)[!.?]?$",
    r"^(hello|hi|hey|good\s*(morning|afternoon|evening))[!.?]?$",
]


def classify(issue_text: str, subject: str = "") -> dict | None:
    """
    Pre-screen a ticket before calling the LLM.
    Returns a result dict if the ticket can be decided without LLM, else returns None.
    
    Return shape:
    {
        "status": "replied" | "escalated",
        "request_type": "product_issue" | "invalid" | ...,
        "product_area": "security" | "fraud" | "out_of_scope" | "general",
        "response": "<canned response>",
        "justification": "<why this decision was made>",
        "trigger": "<which rule fired>"  # internal only, not written to CSV
    }
    """
    combined = (issue_text + " " + subject).lower().strip()

    # 1. Prompt injection — highest priority
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return {
                "status": "escalated",
                "request_type": "invalid",
                "product_area": "security",
                "response": "This request cannot be processed. It has been flagged and escalated to our security team.",
                "justification": "Prompt injection attempt detected. The message contains patterns attempting to override system instructions.",
                "trigger": f"injection:{pattern}"
            }

    # 2. Malicious commands
    for pattern in MALICIOUS_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return {
                "status": "escalated",
                "request_type": "invalid",
                "product_area": "security",
                "response": "This request cannot be fulfilled. It has been flagged for review.",
                "justification": "Malicious or destructive command detected in the support ticket.",
                "trigger": f"malicious:{pattern}"
            }

    # 3. Hard escalation patterns (fraud, identity theft, security vulns, etc.)
    for pattern in ESCALATE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            area = _infer_escalation_area(pattern)
            return {
                "status": "escalated",
                "request_type": "product_issue",
                "product_area": area,
                "response": "Your request has been escalated to a specialist who will assist you shortly. Please do not share any sensitive details in this chat.",
                "justification": f"High-risk issue detected matching escalation criteria: {area}. Requires human review.",
                "trigger": f"escalate:{pattern}"
            }

    # 4. Out of scope
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return {
                "status": "replied",
                "request_type": "invalid",
                "product_area": "out_of_scope",
                "response": "I'm only able to assist with issues related to HackerRank, Claude (Anthropic), or Visa. Your question appears to be outside that scope. Please contact the relevant service directly.",
                "justification": "Ticket is entirely unrelated to the three supported companies or products.",
                "trigger": f"out_of_scope:{pattern}"
            }

    # 5. Trivial / no actionable content
    if re.search(r'|'.join(TRIVIAL_PATTERNS), combined, re.IGNORECASE):
        return {
            "status": "replied",
            "request_type": "invalid",
            "product_area": "general",
            "response": "Thank you for reaching out! If you have a specific issue or question, please describe it and we'll be happy to help.",
            "justification": "Message contains no actionable support request.",
            "trigger": "trivial"
        }

    # 6. Language detection — non-English
    try:
        lang = detect(issue_text)
        if lang != "en":
            return {
                "status": "escalated",
                "request_type": "product_issue",
                "product_area": "general",
                "response": "We have received your message and are escalating it to a team member who can assist in your language.",
                "justification": f"Non-English input detected (language: {lang}). Escalated for human handling.",
                "trigger": f"non_english:{lang}"
            }
    except LangDetectException:
        pass  # short text may fail detection — continue to LLM

    # 7. Vague with no detail — flag but don't pre-decide; let retriever + LLM handle
    if len(issue_text.strip().split()) < 5:
        # Very short — return None to let agent attempt retrieval, 
        # but agent will escalate if retrieval score is below threshold
        pass

    return None  # proceed to LLM pipeline


def _infer_escalation_area(pattern: str) -> str:
    if "fraud" in pattern or "stolen" in pattern or "identity" in pattern:
        return "fraud_and_security"
    if "vulnerability" in pattern or "bug_bounty" in pattern or "exploit" in pattern:
        return "security_vulnerability"
    if "account" in pattern or "unauthorized" in pattern or "hacked" in pattern:
        return "account_security"
    return "general_escalation"
