"""
The one definition of "which intent does this customer or case-worker
question belong to." Plain data — no RedisVL or Postgres import here — so
both engines' routers are built from literally the same routes, the same
reference utterances, and the same distance thresholds. redis_store.py
wraps this into RedisVL Route objects for SemanticRouter; pg_store.py
loads it straight into a Postgres table and does the nearest-reference
math by hand. If the two engines ever disagree on a routing decision, the
question is never "whose route list is right" — there's only one.

Every route's `distance_threshold` is a cosine-distance cutoff (0 =
identical, 2 = opposite) below which a match counts as that route at
all. Below the threshold, both engines answer "no confident match."

`search_target` names which corpus (`"procedures"`, `"sop"`, or `None`)
answers this intent directly. This is the "route to the right vector
search" scenario: once a question is classified, if `search_target` is
set, both engines immediately run that corpus's own semantic search
using the SAME query embedding — no second embedding call, no manual
"which index do I search" logic in application code, just routing
deciding it. `None` means the intent needs a live case record (an open
case's current status, an escalation queue) that no static corpus in
this demo can answer.
"""

ROUTES = [
    {
        "name": "address_update",
        "description": "Customer wants to add or change their mailing or residential address.",
        "action": "Open an address-update case and route it for identity confirmation before the change is applied.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "I need to update my mailing address",
            "I moved and need to change my address on file",
            "How do I update my home address with the bank",
            "My address changed, how do I let you know",
            "I want to change where my statements get mailed",
            "Can you update my address, I just moved to a new state",
            "I need to correct my address, it's wrong on file",
            "How do I change my registered business address",
            "My mail keeps getting returned, I think my address is outdated",
            "I'd like to update my current residential address",
        ],
    },
    {
        "name": "duplicate_tin_resolution",
        "description": "Customer or case worker suspects two customer profiles share the same TIN or SSN.",
        "action": "Open a duplicate-TIN merge review comparing both profiles' identity documents and history.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "I think I have two accounts with the same social security number",
            "There seem to be two profiles under my TIN",
            "My SSN might be linked to someone else's account by mistake",
            "I noticed a duplicate record tied to my tax ID",
            "Two of my accounts share the same TIN, is that a problem",
            "Why do I have two customer profiles with the same SSN",
            "I got a notice about a duplicate TIN on my account",
            "Can you merge my two accounts that share one SSN",
        ],
    },
    {
        "name": "tin_ssn_correction",
        "description": "Customer's TIN or SSN on file is wrong and needs correcting.",
        "action": "Request identity documentation and open a TIN/SSN correction case.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "My social security number is wrong on my account",
            "The SSN on file doesn't match my actual number",
            "I need to correct my tax ID with the bank",
            "My TIN doesn't match what the IRS has on record",
            "How do I fix an incorrect SSN on my account",
            "There's a typo in my social security number on file",
        ],
    },
    {
        "name": "beneficial_ownership_update",
        "description": "Business customer needs to update or recertify beneficial ownership information.",
        "action": "Collect an updated beneficial ownership certification and route it to compliance.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "I need to update the beneficial owners on my business account",
            "Our company's ownership changed, how do I report that",
            "We need to recertify beneficial ownership for our business account",
            "One of our beneficial owners left the company, how do we update that",
            "How do I submit a new beneficial ownership certification",
            "Our business has a new majority owner, what paperwork is needed",
        ],
    },
    {
        "name": "identity_verification_refresh",
        "description": "Customer's identity documents are expired or need re-verification.",
        "action": "Open an identity document refresh case and place the account under a soft restriction until resolved.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "My ID expired, how do I update it with the bank",
            "I need to re-verify my identity for my account",
            "The bank asked me to resubmit my identification",
            "How do I upload a new driver's license for verification",
            "My account says my identity documents need refreshing",
            "I got a notice that my ID on file expired",
        ],
    },
    {
        "name": "due_diligence_refresh",
        "description": "Customer asks about a periodic due-diligence or customer-information review.",
        "action": "Open a due-diligence refresh case and confirm current employment, address, and expected activity.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "Why is the bank asking me to update my employment information again",
            "I got a request for a periodic account review, what's that about",
            "The bank wants updated information about my account activity",
            "What is a due diligence refresh and why am I getting one",
            "I need to complete a customer information update the bank requested",
            "Why does my account need a periodic risk review",
        ],
    },
    {
        "name": "deceased_customer_processing",
        "description": "Customer or family member is reporting a death and needs the account processed.",
        "action": "Restrict the account, notify any joint holders, and open estate processing.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "My father passed away and I need to handle his bank account",
            "How do I report a death and close a family member's account",
            "I have a death certificate for a joint account holder",
            "What do I need to submit to process a deceased customer's account",
            "My mother died, what happens to her accounts now",
            "I'm the executor of an estate, how do I access the account",
        ],
    },
    {
        "name": "name_change",
        "description": "Customer's legal name changed and needs to be updated on file.",
        "action": "Collect supporting legal documentation and open a legal name change case.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "I got married and need to update my last name",
            "How do I change my legal name on my account after a divorce",
            "I have a court order for a name change, how do I submit it",
            "My name changed, what documents do you need",
            "Can I update my name without closing my account",
        ],
    },
    {
        "name": "account_merge_review",
        "description": "Customer or automated matching flags two profiles as likely duplicates.",
        "action": "Compare both profiles and, if confirmed, merge under the older account.",
        "distance_threshold": 0.5,
        "search_target": "sop",
        "references": [
            "I think I accidentally have two customer profiles",
            "Why do I have duplicate accounts under my name",
            "Can you merge my two customer profiles into one",
            "I was flagged as a possible duplicate customer, what does that mean",
            "How do I combine two profiles that are really the same person",
        ],
    },
    {
        "name": "procedure_lookup",
        "description": "Customer or case worker wants to know which procedure applies to a given situation.",
        "action": "Run a semantic search over the procedure catalog and surface the matching procedure.",
        "distance_threshold": 0.5,
        "search_target": "procedures",
        "references": [
            "What's the process for updating a business address",
            "How long does a beneficial ownership update usually take",
            "What procedure applies to a deceased customer's account",
            "Which process do I follow for a duplicate TIN issue",
            "What's the standard process for a due diligence refresh",
            "How do I know which procedure applies to my situation",
            "What's the turnaround time for a name change request",
        ],
    },
    {
        "name": "case_status_inquiry",
        "description": "Customer wants the current status of an already-open case.",
        "action": "Look up the case in the live case system and return its current status.",
        "distance_threshold": 0.5,
        "search_target": None,
        "references": [
            "What's the status of my address change request",
            "Can you check on my open case",
            "How much longer will my due diligence review take",
            "I submitted a request last week, is it done yet",
            "Can you give me an update on my TIN correction case",
            "Where does my case stand right now",
        ],
    },
    {
        "name": "escalation_request",
        "description": "Customer wants an open case escalated to a supervisor or compliance.",
        "action": "Escalate the case to a supervisor-level review.",
        "distance_threshold": 0.5,
        "search_target": None,
        "references": [
            "I want to escalate my case to a supervisor",
            "This has taken too long, can someone else look at it",
            "I need to speak to someone above the person handling my case",
            "My case missed its deadline and I want it escalated",
            "Please escalate this, it's urgent",
        ],
    },
]

ROUTE_NAMES = [r["name"] for r in ROUTES]
