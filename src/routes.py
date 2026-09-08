"""
The one definition of "which intent does this customer question belong
to." Plain data — no RedisVL or Postgres import here — so both engines'
routers are built from literally the same routes, the same reference
utterances, and the same distance thresholds. redis_store.py wraps this
into RedisVL Route objects for SemanticRouter; pg_store.py loads it
straight into a Postgres table and does the nearest-reference math by
hand. If the two engines ever disagree on a routing decision, the
question is never "whose route list is right" — there's only one.

Every route's `distance_threshold` is a cosine-distance cutoff (0 =
identical, 2 = opposite) below which a match counts as that route at
all. Below the threshold, both engines answer "no confident match."
"""

ROUTES = [
    {
        "name": "dispute_transaction",
        "description": "Customer wants to dispute or ask about disputing a charge.",
        "action": "Open a dispute case on the flagged transaction and issue a temporary credit.",
        "distance_threshold": 0.5,
        "references": [
            "I don't recognize this charge on my statement",
            "I want to dispute a transaction",
            "Someone charged me twice for the same purchase",
            "This merchant charged me the wrong amount",
            "I never received the item I ordered but got charged",
            "How do I file a dispute for a charge I didn't make",
            "I was charged for a subscription I canceled",
            "This transaction looks like an error, how do I contest it",
            "I returned this item but never got refunded",
            "Can you help me get my money back for a bad transaction",
            "I want to challenge a charge from a hotel",
            "The amount charged doesn't match my receipt",
        ],
    },
    {
        "name": "report_fraud",
        "description": "Customer suspects fraud, a stolen card, or unauthorized account access.",
        "action": "Freeze the card immediately, open a fraud case, and issue a replacement card.",
        "distance_threshold": 0.5,
        "references": [
            "My card was stolen and I need to report it",
            "I think someone is using my card without permission",
            "I lost my wallet with my credit card in it",
            "There are charges on my account I definitely didn't make",
            "I need to report suspected identity theft",
            "Someone opened a card in my name that isn't mine",
            "My card information may have been compromised in a data breach",
            "I got a text about a purchase I never made",
            "Please freeze my card right now, it might be stolen",
            "I want to report unauthorized access to my account",
        ],
    },
    {
        "name": "check_balance",
        "description": "Customer wants their current balance or available credit.",
        "action": "Return the current balance and available credit for the customer's card.",
        "distance_threshold": 0.5,
        "references": [
            "What is my current credit card balance",
            "How much do I owe on my card right now",
            "What's my available credit",
            "Can you tell me my account balance",
            "How much credit do I have left this month",
            "What do I currently owe",
            "Show me my balance",
            "How close am I to my credit limit right now",
        ],
    },
    {
        "name": "payment_due_date",
        "description": "Customer asks about when a payment is due or payment scheduling.",
        "action": "Return the next payment due date and minimum payment amount.",
        "distance_threshold": 0.5,
        "references": [
            "When is my credit card payment due",
            "What day of the month do I need to pay by",
            "Can I change my payment due date",
            "How much is my minimum payment this month",
            "When will my next bill be due",
            "I want to set up automatic payments",
            "Can I schedule a payment for next week",
            "What happens if I pay a day late",
        ],
    },
    {
        "name": "credit_limit_increase",
        "description": "Customer wants to request or ask about a higher credit limit.",
        "action": "Run a soft credit check and route to the credit-limit-increase workflow.",
        "distance_threshold": 0.5,
        "references": [
            "I want to request a credit limit increase",
            "Can you raise my credit limit",
            "How do I get a higher spending limit on my card",
            "My limit feels too low, can it be increased",
            "Will asking for a higher limit hurt my credit score",
            "I need more available credit for an upcoming purchase",
            "Why was my credit limit reduced",
        ],
    },
    {
        "name": "card_recommendation",
        "description": "Customer wants help choosing which credit card product fits their spending.",
        "action": "Run a semantic search over the card product catalog and recommend the best match.",
        "distance_threshold": 0.5,
        "references": [
            "Which card is best for someone who travels a lot",
            "I'm looking for a card with no annual fee",
            "What's the best card for grocery and restaurant spending",
            "I want a card with good cash back on gas",
            "Which of your cards has the best travel rewards",
            "I'm a student, which card should I get",
            "I need a card to help rebuild my credit",
            "What card would you recommend for a small business",
            "Is there a card with a low interest rate for carrying a balance",
            "Which card has airport lounge access",
        ],
    },
    {
        "name": "rewards_question",
        "description": "Customer asks how rewards, points, or cash back work on an existing card.",
        "action": "Return the rewards program details for the customer's current card product.",
        "distance_threshold": 0.5,
        "references": [
            "How do my cash back rewards get paid out",
            "Do my points expire",
            "How do I redeem my rewards points",
            "Can I transfer my points to an airline",
            "How much cash back have I earned this year",
            "What categories earn extra rewards on my card",
        ],
    },
    {
        "name": "apr_question",
        "description": "Customer asks about interest rate, APR, or how interest is calculated.",
        "action": "Return the customer's current APR and a plain-language explanation of how interest accrues.",
        "distance_threshold": 0.5,
        "references": [
            "What is my current interest rate",
            "How is interest calculated on my balance",
            "Why did my APR go up",
            "What does APR actually mean",
            "If I pay in full do I still get charged interest",
            "Is there a way to lower my interest rate",
        ],
    },
    {
        "name": "foreign_transaction_fee",
        "description": "Customer asks about fees for using their card abroad or in a foreign currency.",
        "action": "Return whether the customer's card charges a foreign transaction fee and the rate.",
        "distance_threshold": 0.5,
        "references": [
            "Does my card charge a fee for purchases made abroad",
            "I'm traveling internationally, will I be charged extra",
            "What's the foreign transaction fee on this card",
            "Is there a fee for using my card in a foreign currency",
            "Which of your cards has no foreign transaction fee",
        ],
    },
    {
        "name": "close_account",
        "description": "Customer wants to close their credit card account.",
        "action": "Confirm the balance is paid in full, then route to account closure.",
        "distance_threshold": 0.5,
        "references": [
            "I want to close my credit card account",
            "How do I cancel this card",
            "I don't want this card anymore, how do I close it",
            "Will closing my account hurt my credit score",
            "Can I cancel my card if I still have a balance",
        ],
    },
    {
        "name": "statement_question",
        "description": "Customer has a question about reading or receiving their statement.",
        "action": "Explain the statement cycle and offer to enable paper or electronic statements.",
        "distance_threshold": 0.5,
        "references": [
            "When is my statement generated each month",
            "Why is my statement balance different from my current balance",
            "Can I get paper statements mailed to me",
            "How far back can I view old statements",
            "I didn't get my statement this month",
        ],
    },
    {
        "name": "authorized_user",
        "description": "Customer asks about adding or removing an authorized user on their account.",
        "action": "Start the authorized-user add/remove workflow on the customer's account.",
        "distance_threshold": 0.5,
        "references": [
            "How do I add an authorized user to my account",
            "Can my spouse get a card on my account",
            "How do I remove an authorized user",
            "Is an authorized user responsible for the balance",
            "Can my teenager have a card linked to my account",
        ],
    },
]

ROUTE_NAMES = [r["name"] for r in ROUTES]
