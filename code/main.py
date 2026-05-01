import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from retriever import build_index
from agent import triage
from config import TICKETS_PATH, OUTPUT_PATH, LOG_PATH

# Match sample_support_tickets.csv column names exactly
OUTPUT_COLUMNS = ["Issue", "Subject", "Company", "Response", "Product Area", "Status", "Request Type", "Justification"]
# Internal keys from agent output
AGENT_KEYS = {"response": "Response", "product_area": "Product Area", "status": "Status", "request_type": "Request Type", "justification": "Justification"}


def main():
    print("=" * 60)
    print("HackerRank Orchestrate — Support Triage Agent")
    print("=" * 60)
    print(f"Log file: {LOG_PATH}")
    print()

    # Step 1: Build (or load) the corpus index
    print("[1/3] Building corpus index...")
    build_index(force_rebuild=False)
    print()

    # Step 2: Load tickets
    print(f"[2/3] Loading tickets from {TICKETS_PATH}...")
    if not TICKETS_PATH.exists():
        print(f"ERROR: {TICKETS_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(TICKETS_PATH)
    print(f"      Loaded {len(df)} tickets.")
    print()

    # Step 3: Process each ticket
    print("[3/3] Processing tickets...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write header row
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(OUTPUT_COLUMNS) + "\n")

    stats = {"replied": 0, "escalated": 0, "errors": 0}
    import time

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Triaging"):
        try:
            result = triage(row.to_dict())
            stats[result["status"]] += 1
        except Exception as e:
            print(f"\n[main] ERROR on row {idx}: {e}")
            result = {
                "status": "escalated",
                "product_area": "general",
                "response": "An error occurred while processing this ticket. A support agent will follow up.",
                "justification": f"Processing error: {str(e)[:100]}",
                "request_type": "product_issue"
            }
            stats["errors"] += 1

        # Build output row matching sample format:
        # Issue, Subject, Company, Response, Product Area, Status, Request Type
        out_row = {
            "Issue": str(row.get("Issue", "")).replace('"', "'").replace("\n", " "),
            "Subject": str(row.get("Subject", "")).replace('"', "'").replace("\n", " "),
            "Company": str(row.get("Company", "")).replace('"', "'").replace("\n", " "),
            "Response": str(result.get("response", "")).replace('"', "'").replace("\n", " "),
            "Product Area": str(result.get("product_area", "")).replace('"', "'").replace("\n", " "),
            "Status": str(result.get("status", "")).replace('"', "'").replace("\n", " ").capitalize(),
            "Request Type": str(result.get("request_type", "")).replace('"', "'").replace("\n", " "),
            "Justification": str(result.get("justification", "")).replace('"', "'").replace("\n", " "),
        }

        row_values = [out_row[col] for col in OUTPUT_COLUMNS]
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(",".join([f'"{v}"' for v in row_values]) + "\n")

        # Small delay — key rotation handles rate limits
        time.sleep(2)

    print()
    print("=" * 60)
    print(f"DONE. Results written to: {OUTPUT_PATH}")
    print(f"  Replied:   {stats['replied']}")
    print(f"  Escalated: {stats['escalated']}")
    print(f"  Errors:    {stats['errors']}")
    print(f"  Log:       {LOG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
