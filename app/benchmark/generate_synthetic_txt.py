import argparse
from pathlib import Path
import random


OUTPUT_DIR = Path("data/benchmark/raw_txt")


DOMAINS = [
    "telecom",
    "database",
    "infrastructure",
    "cloud",
    "network",
    "security",
]

SYSTEMS = {
    "telecom": ["Voucher", "Billing", "SMSC", "IN", "Charging"],
    "database": ["Oracle", "PostgreSQL", "MySQL"],
    "infrastructure": ["Linux", "Windows", "Storage"],
    "cloud": ["Kubernetes", "Docker", "API Gateway"],
    "network": ["Router", "Firewall", "Load Balancer"],
    "security": ["IAM", "WAF", "SIEM"],
}

ISSUES = [
    "connection timeout",
    "authentication failure",
    "high latency",
    "service unavailable",
    "transaction failure",
    "disk utilization high",
    "memory usage high",
    "thread pool saturation",
    "database listener error",
    "message delivery failure",
    "voucher recharge failure",
    "pod restart loop",
    "certificate expiry",
    "request did not reach platform",
]

ACTIONS = [
    "check application logs",
    "verify network connectivity",
    "validate service configuration",
    "restart impacted service after approval",
    "check database session count",
    "review error code and timestamp",
    "compare with previous RCA",
    "check downstream dependency health",
    "validate retry configuration",
    "review capacity trend",
]


def build_document(index: int) -> str:
    # Deterministic but unique per file.
    random.seed(index)

    domain = random.choice(DOMAINS)
    system = random.choice(SYSTEMS[domain])
    issue = random.choice(ISSUES)
    action_list = random.sample(ACTIONS, 4)

    incident_id = f"SYN-INC-{index:05d}"
    ticket_id = f"TKT-{100000 + index}"
    node_name = f"{system.lower().replace(' ', '-')}-node-{index:05d}"
    trace_id = f"TRACE-{index:05d}-{domain.upper()}-{system.upper().replace(' ', '-')}"
    customer_ref = f"CUST-{index % 250:03d}"
    event_time = f"2026-06-{(index % 28) + 1:02d} {(index % 24):02d}:{(index % 60):02d}:00"

    return f"""
Title: Synthetic Industrial Incident {index}
Domain: {domain}
System: {system}
Document Type: Incident
Incident ID: {incident_id}
Ticket ID: {ticket_id}
Node Name: {node_name}
Trace ID: {trace_id}
Customer Reference: {customer_ref}
Event Time: {event_time}

Incident Summary:
The {system} system reported {issue}. The issue affected operational processing and required investigation by the support team.
This unique synthetic incident is identified by incident id {incident_id}, ticket id {ticket_id}, trace id {trace_id}, and node {node_name}.

Symptoms:
Users observed intermittent failures, delayed responses, and inconsistent transaction behavior.
Monitoring dashboards showed alerts related to {issue}.
The event was observed at {event_time} for customer reference {customer_ref}.

Investigation:
The engineer should {action_list[0]}, {action_list[1]}, {action_list[2]}, and {action_list[3]}.
The investigation should correlate logs using trace id {trace_id} and node name {node_name}.

Resolution:
After identifying the root cause, apply the approved remediation step, validate service health, confirm transaction recovery, and update ticket {ticket_id}.

Root Cause:
The most likely cause was related to {issue} in the {system} platform for node {node_name}.

Preventive Action:
Add monitoring threshold, improve alerting, update SOP, and review similar incidents from historical knowledge base.
Reference this case as {incident_id} for future benchmark retrieval testing.
""".strip()


def generate_documents(count: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for index in range(1, count + 1):
        file_path = OUTPUT_DIR / f"synthetic_incident_{index:05d}.txt"

        if file_path.exists():
            skipped += 1
            continue

        content = build_document(index)
        file_path.write_text(content, encoding="utf-8")
        created += 1

    print(f"Target synthetic TXT files: {count}")
    print(f"Created new files: {created}")
    print(f"Already existed/skipped: {skipped}")
    print(f"Output directory: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic industrial TXT files")
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Target number of synthetic TXT files to keep in benchmark folder",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be greater than 0")

    generate_documents(count=args.count)


if __name__ == "__main__":
    main()