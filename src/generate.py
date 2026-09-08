"""
Generates the whole synthetic corpus: customers, the operational requests
filed against their identity records, the procedure catalog case workers
follow, and a set of standard-operating-procedure (SOP) knowledge base
articles.

Every name and case note below is fabricated. Nothing here is drawn from
or resembles a real institution's data or a real customer's data. The
scenario is modeled on a bank's customer identity operations team — the
group that keeps customer identity records accurate and compliant: address
updates, duplicate TIN/SSN resolution, and related customer-data
maintenance and due-diligence work — but the bank, its customers, and
every case are invented for this demo.
"""

import datetime
import json
import os
import random

from config import CUSTOMERS, DATA_DIR, DATA_FILES, PROCEDURES, REQUESTS, SEED, SOP_ARTICLES

rng = random.Random(SEED)

# Anchor date for "N days ago" generation below — fixed rather than
# datetime.date.today() so the corpus is reproducible run to run.
TODAY = datetime.date(2026, 9, 8)

CITIES = [
    ("New York", "NY"), ("Charlotte", "NC"), ("Chicago", "IL"), ("Dallas", "TX"),
    ("San Francisco", "CA"), ("Minneapolis", "MN"), ("Boston", "MA"), ("Denver", "CO"),
    ("Atlanta", "GA"), ("Phoenix", "AZ"), ("Seattle", "WA"), ("Miami", "FL"),
    ("Philadelphia", "PA"), ("St. Louis", "MO"), ("Portland", "OR"), ("Nashville", "TN"),
    ("Salt Lake City", "UT"), ("San Diego", "CA"), ("Columbus", "OH"), ("Indianapolis", "IN"),
]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Linda", "Michael", "Barbara",
    "David", "Elizabeth", "William", "Jennifer", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Karen", "Charles", "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony",
    "Sandra", "Mark", "Ashley", "Donald", "Kimberly", "Steven", "Emily", "Paul", "Donna",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
]

# Each request type carries its own intake-channel pool and a handful of
# case-note templates, so embeddings cluster meaningfully by type — the
# whole point of a vector search demo is that "address change requests"
# and "duplicate TIN cases" land in genuinely different neighborhoods of
# the vector space, not that the text is realistic case-management prose.
REQUEST_TYPES = {
    "Address Update": {
        "channels": ["Branch - Downtown", "Online Portal", "Phone Banking", "Relationship Manager", "Mail"],
        "templates": [
            "Customer submitted a change of mailing address via {m}",
            "Updated home address on file following a move, submitted via {m}",
            "Address correction requested through {m} after mail was returned undeliverable",
        ],
    },
    "Duplicate TIN Resolution": {
        "channels": ["Compliance Queue", "Data Quality Review", "Branch - Downtown", "Back Office Ops"],
        "templates": [
            "Two customer profiles found sharing the same TIN, flagged via {m} for merge review",
            "Duplicate SSN detected between two open accounts during {m} review",
            "TIN conflict identified by {m}, pending resolution",
        ],
    },
    "TIN/SSN Correction": {
        "channels": ["Online Portal", "Branch - Downtown", "Phone Banking", "Mail"],
        "templates": [
            "Customer reported an incorrect SSN on file, correction requested via {m}",
            "TIN mismatch with IRS records found during {m} review, correction submitted",
        ],
    },
    "Beneficial Ownership Update": {
        "channels": ["Relationship Manager", "Compliance Queue", "Branch - Downtown"],
        "templates": [
            "Business customer submitted updated beneficial ownership information via {m}",
            "Change in beneficial owner reported through {m}, documentation pending",
        ],
    },
    "Identity Verification Refresh": {
        "channels": ["Online Portal", "Branch - Downtown", "Phone Banking"],
        "templates": [
            "Periodic identity verification refresh initiated via {m}",
            "Customer asked to re-verify identity documents through {m} after an expired ID",
        ],
    },
    "Due Diligence Refresh": {
        "channels": ["Compliance Queue", "Relationship Manager", "Back Office Ops"],
        "templates": [
            "Scheduled due-diligence refresh review opened via {m}",
            "Enhanced due diligence triggered by {m} following a risk-rating change",
        ],
    },
    "Deceased Customer Processing": {
        "channels": ["Branch - Downtown", "Phone Banking", "Mail"],
        "templates": [
            "Next of kin notified the bank of customer's passing via {m}, estate processing opened",
            "Death certificate received through {m}, account flagged for estate handling",
        ],
    },
    "Name Change": {
        "channels": ["Branch - Downtown", "Online Portal", "Mail"],
        "templates": [
            "Customer requested a legal name change update via {m} following marriage",
            "Name correction submitted through {m} with supporting legal documentation",
        ],
    },
    "Account/Profile Merge Review": {
        "channels": ["Data Quality Review", "Compliance Queue", "Back Office Ops"],
        "templates": [
            "Two customer profiles flagged as possible duplicates by {m}, pending merge review",
            "Automated matching flagged a possible duplicate profile via {m}",
        ],
    },
    "Tax Document Correction": {
        "channels": ["Online Portal", "Phone Banking", "Mail"],
        "templates": [
            "Customer requested a corrected 1099 due to a TIN error, submitted via {m}",
            "Tax form reissue requested through {m} after a name mismatch",
        ],
    },
    "Case Status Inquiry": {
        "channels": ["Phone Banking", "Online Portal", "Branch - Downtown"],
        "templates": [
            "Customer called via {m} asking for an update on an open address change case",
            "Status check on an open due-diligence case requested through {m}",
        ],
    },
    "Escalation Request": {
        "channels": ["Relationship Manager", "Compliance Queue", "Phone Banking"],
        "templates": [
            "Customer asked to escalate an unresolved TIN dispute via {m}",
            "Case escalated to compliance by {m} after missed SLA",
        ],
    },
}
REQUEST_TYPE_NAMES = list(REQUEST_TYPES.keys())

# Category taxonomy is shared between the procedure catalog and the SOP
# knowledge base (the same rough groupings a case worker would file both
# under), but each corpus's rows and descriptions are independent.
PROCEDURE_CATALOG = [
    {"name": "Standard Address Change - Individual Customer", "category": "Address Maintenance", "sla_days": 1,
     "description": "Update a mailing or residential address for an individual customer once submitted through any channel with the customer's identity confirmed. No supporting documentation required unless the new address is outside the customer's stated state of residence, in which case a secondary ID check is triggered."},
    {"name": "Address Change - Business Customer", "category": "Address Maintenance", "sla_days": 3,
     "description": "Update the registered or mailing address for a business account. Requires confirmation from an authorized signer and, for a change of registered agent address, an updated formation document on file."},
    {"name": "Duplicate TIN Merge Review", "category": "TIN/SSN Resolution", "sla_days": 10,
     "description": "Resolve two customer profiles that share the same TIN or SSN. Compliance reviews both profiles' request history and identity documents before deciding whether to merge the records or correct one profile's TIN."},
    {"name": "TIN/SSN Correction", "category": "TIN/SSN Resolution", "sla_days": 5,
     "description": "Correct a customer's TIN or SSN on file after a mismatch with IRS or SSA records is identified. Requires a copy of the customer's Social Security card or IRS-issued TIN confirmation letter."},
    {"name": "Beneficial Ownership Certification Update", "category": "Beneficial Ownership", "sla_days": 15,
     "description": "Collect and file an updated beneficial ownership certification for a business account after a change in ownership structure, in line with the bank's customer due-diligence program."},
    {"name": "Identity Document Refresh", "category": "Identity Verification", "sla_days": 7,
     "description": "Re-verify a customer's government-issued identification after it expires or is flagged during a periodic review. The account is placed under a soft restriction until a valid document is received."},
    {"name": "Enhanced Due Diligence Review", "category": "Due Diligence", "sla_days": 30,
     "description": "A deeper review triggered by a risk-rating change, high-risk jurisdiction activity, or a periodic schedule for higher-risk customer segments. Includes a source-of-funds review and a supervisor sign-off before the account's risk rating is updated."},
    {"name": "Standard Due-Diligence Refresh", "category": "Due Diligence", "sla_days": 20,
     "description": "The routine periodic refresh of a customer's due-diligence file — confirming current employment, address, and expected account activity remain consistent with the customer's risk profile."},
    {"name": "Deceased Customer Account Processing", "category": "Estate Processing", "sla_days": 20,
     "description": "Process the accounts of a deceased customer once a death certificate is received: restrict the account from further transactions, notify any joint holders, and route remaining balances per the estate's instructions or applicable state law."},
    {"name": "Legal Name Change Processing", "category": "Documentation", "sla_days": 5,
     "description": "Update a customer's legal name on file after marriage, divorce, or a court order, once supporting legal documentation is received and matched against the customer's identity file."},
    {"name": "Duplicate Profile Merge Review", "category": "Documentation", "sla_days": 10,
     "description": "Review two customer profiles flagged as likely duplicates by automated matching. If confirmed, the profiles are merged under the older account and the duplicate is closed with a note in the audit trail."},
    {"name": "Tax Document Reissue", "category": "Documentation", "sla_days": 5,
     "description": "Reissue a 1099 or other tax document after a correction to the customer's name or TIN, once the underlying identity record has already been corrected."},
    {"name": "Compliance Escalation Review", "category": "Escalations", "sla_days": 3,
     "description": "A supervisor-level review for a case that missed its original service-level target or where the customer has asked to escalate beyond front-line staff."},
    {"name": "Address Verification Hold Release", "category": "Address Maintenance", "sla_days": 2,
     "description": "Release a temporary hold placed on an account after an address change to a high-risk region, once a secondary identity check confirms the change is legitimate."},
    {"name": "Joint Account Holder Identity Update", "category": "Identity Verification", "sla_days": 7,
     "description": "Update the identity record for a secondary or joint account holder, including a fresh identity verification if their information was never independently confirmed at account opening."},
]

SOP_CATALOG = [
    ("Address Changes", "How do I update a customer's mailing address?",
     "Address updates can be submitted through any channel — branch, phone, online portal, or mail — once the customer's identity has been confirmed. Standard changes complete within one business day; changes to a residential address outside the customer's stated state of residence trigger a secondary identity check first."),
    ("Address Changes", "What happens if a customer's mail is returned as undeliverable?",
     "An undeliverable-mail flag opens an address-verification case automatically. The account is not restricted, but statements switch to electronic-only until a confirmed current address is on file."),
    ("Address Changes", "Can a business customer update its registered address online?",
     "No — a business address change requires confirmation from an authorized signer and, if the registered agent address is changing, an updated formation document. Route it to a relationship manager rather than the self-service portal."),
    ("TIN/SSN Issues", "What do I do if two customer profiles share the same TIN?",
     "Open a Duplicate TIN Merge Review case. Compliance compares both profiles' identity documents and account history before deciding whether to merge the records into one profile or correct the TIN on one of them."),
    ("TIN/SSN Issues", "A customer says their SSN is wrong on file — what's the process?",
     "Request a copy of their Social Security card or an IRS TIN confirmation letter, then submit a TIN/SSN Correction case. The correction typically completes within 5 business days once the document is received."),
    ("TIN/SSN Issues", "How long does a duplicate TIN resolution usually take?",
     "The service-level target is 10 business days, though cases involving a full profile merge (rather than a single-field correction) can take longer if account history needs to be reconciled first."),
    ("Beneficial Ownership", "When does a business customer need to recertify beneficial ownership?",
     "Whenever there's a change in who owns 25% or more of the business, or who exercises significant control over it, and at minimum during the periodic due-diligence refresh cycle for that account."),
    ("Beneficial Ownership", "What documentation is required for a beneficial ownership update?",
     "A completed beneficial ownership certification form plus identity documentation for each newly reported beneficial owner. The service-level target is 15 business days."),
    ("Beneficial Ownership", "Can beneficial ownership updates be submitted online?",
     "Not currently — route these to a relationship manager or the compliance queue, since the certification form requires an authorized signer's signature."),
    ("Due Diligence", "What triggers an enhanced due-diligence review?",
     "A risk-rating change, activity in a high-risk jurisdiction, or reaching the periodic review point for a customer segment already rated higher risk. It includes a source-of-funds review and a supervisor sign-off."),
    ("Due Diligence", "How is a standard due-diligence refresh different from enhanced due diligence?",
     "A standard refresh simply confirms that employment, address, and expected account activity still match the customer's existing risk profile. Enhanced due diligence goes further, examining source of funds, and requires supervisor sign-off before the file closes."),
    ("Due Diligence", "What happens if a customer doesn't respond to a due-diligence refresh request?",
     "The case stays open past its service-level target and escalates to a supervisor review. Accounts are not restricted solely for a slow refresh response unless a separate risk flag is also present."),
    ("Identity Verification", "What happens when a customer's ID expires?",
     "The account is placed under a soft restriction — existing activity continues, but certain self-service changes are paused — until a current, valid identification document is received and an Identity Document Refresh case is closed."),
    ("Identity Verification", "Does a joint account holder need to independently verify their identity?",
     "Yes, if their information was never independently confirmed at account opening. Route this as a Joint Account Holder Identity Update case rather than treating it as part of the primary holder's file."),
    ("Identity Verification", "How often is identity verification refreshed for an existing customer?",
     "On the same periodic schedule as the customer's due-diligence refresh, unless an expired ID or a flagged review triggers it sooner."),
    ("Estate Processing", "What's the first step when a customer has passed away?",
     "Once a death certificate is received through any channel, open a Deceased Customer Account Processing case immediately — this restricts the account from further transactions and notifies any joint holders before anything else happens."),
    ("Estate Processing", "Can a joint account holder still use the account after the other holder's death?",
     "Typically yes, once the death certificate is on file and the case is processed — the surviving joint holder retains access, but the deceased customer's individual identity record is closed."),
    ("Estate Processing", "How are remaining balances handled for a deceased customer with no joint holder?",
     "Balances route according to the estate's instructions once provided by the executor, or per applicable state law if no instructions are on file. This case type has a 20 business day service-level target."),
    ("Documentation", "What supporting documents are needed for a legal name change?",
     "A marriage certificate, divorce decree, or court order, matched against the customer's existing identity file. The service-level target is 5 business days."),
    ("Documentation", "How do I know if two customer profiles are true duplicates or just similar?",
     "Automated matching flags likely duplicates based on name, TIN, and address overlap, but a person still reviews both profiles before merging. If confirmed, the older account is kept and the duplicate is closed with a note in the audit trail."),
    ("Documentation", "Why would a customer need a corrected 1099?",
     "Usually because their name or TIN was wrong on the original form. The underlying identity record has to be corrected first — the tax document reissue is a downstream step, not a fix on its own."),
    ("Documentation", "What happens to the closed profile after a duplicate merge?",
     "It stays in the system as a closed record with a note pointing to the surviving profile, so historical activity remains traceable for audit purposes even though the account itself is no longer active."),
    ("Escalations", "When should a case be escalated to compliance?",
     "When it's missed its original service-level target, or the customer explicitly asks to speak with someone beyond front-line staff. Escalations get a supervisor-level review within 3 business days."),
    ("Escalations", "What's different about an escalated case versus a normal one?",
     "A supervisor reviews the full case history and either resolves it directly or reassigns it with clearer instructions. The customer should also get a status update explaining the delay, not just a resolution."),
    ("Escalations", "Can a customer request an escalation before the service-level target has passed?",
     "Yes — the target is a maximum, not a promise that nothing can move faster. If there's a clear reason (a wire deadline, a closing date), note it on the case and flag it for early review."),
    ("Case Management", "How can a customer check the status of an open case?",
     "Through the online portal if they have a case reference number, or by phone banking, which can look up any open case tied to their identity record without needing the reference number."),
]


def rng_pick(seq):
    return seq[rng.randrange(len(seq))]


def rand_date(start_days_ago, end_days_ago):
    day_offset = rng.randint(end_days_ago, start_days_ago)
    return (TODAY - datetime.timedelta(days=day_offset)).isoformat()


def generate_customers():
    customers = []
    for i in range(CUSTOMERS):
        city, state = rng_pick(CITIES)
        customers.append({
            "customer_id": f"C{i:06d}",
            "name": f"{rng_pick(FIRST_NAMES)} {rng_pick(LAST_NAMES)}",
            "city": city,
            "state": state,
            "join_date": rand_date(1400, 30),
        })
    return customers


def generate_procedures():
    procedures = []
    for i, p in enumerate(PROCEDURE_CATALOG[:PROCEDURES]):
        procedures.append({
            "procedure_id": f"P{i:03d}",
            "name": p["name"],
            "category": p["category"],
            "sla_days": p["sla_days"],
            "description": p["description"],
        })
    return procedures


def generate_requests(customers):
    requests = []
    for i in range(REQUESTS):
        customer = rng_pick(customers)
        request_type = rng_pick(REQUEST_TYPE_NAMES)
        rt = REQUEST_TYPES[request_type]
        channel = rng_pick(rt["channels"])
        template = rng_pick(rt["templates"])
        description = template.format(m=channel)
        days_open = rng.randint(0, 45)
        requests.append({
            "request_id": f"R{i:08d}",
            "customer_id": customer["customer_id"],
            "channel": channel,
            "category": request_type,
            "days_open": days_open,
            "date": rand_date(180, 0),
            "description": description,
        })
    return requests


def generate_sop_articles():
    articles = []
    for i, (category, title, body) in enumerate(SOP_CATALOG[:SOP_ARTICLES]):
        articles.append({
            "sop_id": f"SOP{i:03d}",
            "category": category,
            "title": title,
            "body": body,
        })
    return articles


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating customers...")
    customers = generate_customers()
    write_jsonl(DATA_FILES["customers"], customers)
    print(f"  {len(customers):,} customers")

    print("Generating procedure catalog...")
    procedures = generate_procedures()
    write_jsonl(DATA_FILES["procedures"], procedures)
    print(f"  {len(procedures):,} procedures")

    print("Generating requests...")
    requests = generate_requests(customers)
    write_jsonl(DATA_FILES["requests"], requests)
    print(f"  {len(requests):,} requests")

    print("Generating SOP articles...")
    sops = generate_sop_articles()
    write_jsonl(DATA_FILES["sop_articles"], sops)
    print(f"  {len(sops):,} SOP articles")


if __name__ == "__main__":
    main()
