"""
Generates the whole synthetic corpus: customers, their credit cards,
transactions on those cards, the credit card product catalog, and a set of
support FAQ articles.

Every name, merchant, and note below is fabricated. Nothing here is drawn
from or resembles a real institution's data or a real customer's data.
"""

import datetime
import json
import os
import random

from config import CARD_PRODUCTS, CARDS_PER_CUSTOMER_RANGE, CUSTOMERS, DATA_DIR, DATA_FILES, FAQ_ARTICLES, SEED, TRANSACTIONS

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

# Each category carries its own merchant pool and a handful of description
# templates, so embeddings cluster meaningfully by category — the whole
# point of a vector search demo is that "coffee shop purchases" and "hotel
# charges" land in genuinely different neighborhoods of the vector space,
# not that the text is realistic prose.
CATEGORIES = {
    "Coffee Shops": {
        "merchants": ["Blue Bottle Coffee", "Corner Cafe", "Daily Grind Coffee", "Roast House", "Steam & Bean"],
        "templates": ["Coffee and pastry at {m}", "Morning coffee run at {m}", "Latte and breakfast sandwich at {m}"],
    },
    "Dining": {
        "merchants": ["Olive Branch Bistro", "Golden Wok", "Riverside Grill", "Taco Verde", "The Hungry Fork"],
        "templates": ["Dinner for two at {m}", "Lunch order at {m}", "Weekend brunch at {m}"],
    },
    "Groceries": {
        "merchants": ["Green Valley Market", "Sunrise Grocers", "Metro Foods", "Harvest Basket", "Corner Pantry"],
        "templates": ["Weekly grocery shopping at {m}", "Produce and household items at {m}", "Grocery pickup order at {m}"],
    },
    "Travel - Airlines": {
        "merchants": ["Skyline Airlines", "Continental Wings", "Horizon Air", "Blue Sky Airways"],
        "templates": ["Round-trip flight booked on {m}", "Airline ticket change fee, {m}", "Checked bag fee, {m}"],
    },
    "Travel - Hotels": {
        "merchants": ["Harborview Hotel", "Cascade Inn & Suites", "Union Square Hotel", "Lakeside Resort"],
        "templates": ["Two-night hotel stay at {m}", "Hotel room and resort fee, {m}", "Extended stay at {m}"],
    },
    "Ride Share": {
        "merchants": ["QuickRide", "CityHail", "GoCar"],
        "templates": ["Ride from airport via {m}", "Evening ride home via {m}", "Cross-town ride via {m}"],
    },
    "Streaming & Subscriptions": {
        "merchants": ["StreamVue", "MelodyStream", "CloudBox Storage", "NewsDaily Digital"],
        "templates": ["Monthly subscription charge, {m}", "Annual renewal, {m}", "Streaming plan upgrade, {m}"],
    },
    "Utilities": {
        "merchants": ["Metro Power & Light", "CityWater Utility", "Northgate Gas Co."],
        "templates": ["Monthly utility bill, {m}", "Autopay utility payment, {m}"],
    },
    "Online Shopping": {
        "merchants": ["ShopWave", "BrightBasket Online", "QuickCart"],
        "templates": ["Online order from {m}", "Household goods order, {m}", "Return and exchange, {m}"],
    },
    "Electronics": {
        "merchants": ["CircuitPoint Electronics", "ByteHouse", "GadgetWorks"],
        "templates": ["Laptop accessory purchase at {m}", "Home electronics purchase at {m}"],
    },
    "Home Improvement": {
        "merchants": ["Hammer & Nail Supply", "GreenThumb Garden Center", "BuildRight Hardware"],
        "templates": ["Home repair supplies at {m}", "Garden and patio purchase at {m}"],
    },
    "Gas Stations": {
        "merchants": ["Summit Fuel", "Roadway Gas & Go", "Pinnacle Petroleum"],
        "templates": ["Fuel fill-up at {m}", "Gas station convenience purchase at {m}"],
    },
    "Pharmacy & Health": {
        "merchants": ["Wellness Corner Pharmacy", "CareFirst Drugstore", "HealthMart Pharmacy"],
        "templates": ["Prescription pickup at {m}", "Over-the-counter purchase at {m}"],
    },
    "Entertainment": {
        "merchants": ["Downtown Cinema", "Riverside Theater", "GameZone Arcade"],
        "templates": ["Movie tickets at {m}", "Weekend outing at {m}"],
    },
    "Fitness": {
        "merchants": ["PeakForm Gym", "Riverside Yoga Studio", "IronWorks Fitness"],
        "templates": ["Monthly gym membership, {m}", "Drop-in class fee, {m}"],
    },
}
CATEGORY_NAMES = list(CATEGORIES.keys())

CARD_PRODUCT_CATALOG = [
    {"name": "Everyday Cash Rewards Card", "category": "Cash Back", "annual_fee": 0, "apr_range": "18.99%-27.99%",
     "description": "Earn 3% cash back on groceries and dining, 1% on everything else, with no annual fee. A solid everyday card for households that spend most of their budget on groceries and restaurants and don't want to track rotating categories."},
    {"name": "Travel Elite Card", "category": "Travel", "annual_fee": 395, "apr_range": "19.99%-26.99%",
     "description": "Premium travel rewards card earning 5x points on flights and hotels booked through our travel portal, airport lounge access, and no foreign transaction fees. Best suited for frequent travelers who can offset the annual fee with lounge access and travel credits."},
    {"name": "Student Starter Card", "category": "Student", "annual_fee": 0, "apr_range": "22.99%-29.99%",
     "description": "A first credit card for students building credit history, with a low starting credit limit, free credit score monitoring, and cash back on streaming subscriptions and food delivery. No annual fee and no credit history required to apply."},
    {"name": "Business Rewards Card", "category": "Business", "annual_fee": 95, "apr_range": "18.99%-25.99%",
     "description": "Designed for small business owners: 2% cash back on office supplies and software subscriptions, employee cards at no extra cost, and expense-tracking tools built into the mobile app."},
    {"name": "Low Rate Card", "category": "Low Rate", "annual_fee": 0, "apr_range": "14.99%-19.99%",
     "description": "One of our lowest ongoing APRs, aimed at cardholders who sometimes carry a balance month to month rather than paying in full. No rewards program, no annual fee, and a 0% introductory APR on balance transfers for the first 15 months."},
    {"name": "Premium Travel Card", "category": "Travel", "annual_fee": 550, "apr_range": "19.99%-26.99%",
     "description": "Our top-tier travel card: 10x points on hotels and rental cars, a yearly travel credit that offsets the annual fee, airport lounge membership, and trip delay and baggage insurance included."},
    {"name": "Secured Credit Builder Card", "category": "Building Credit", "annual_fee": 0, "apr_range": "24.99%",
     "description": "A secured card backed by a refundable security deposit, built for customers rebuilding or establishing credit. Reports to all three credit bureaus monthly and automatically reviews for an unsecured upgrade after 12 months of on-time payments."},
    {"name": "Grocery & Dining Rewards Card", "category": "Cash Back", "annual_fee": 0, "apr_range": "18.99%-27.99%",
     "description": "4% cash back at grocery stores and restaurants, 1% on everything else. No annual fee and no rotating categories to track — the two categories that earn extra are fixed."},
    {"name": "Gas Rewards Card", "category": "Cash Back", "annual_fee": 0, "apr_range": "19.99%-27.99%",
     "description": "5% cash back at gas stations, 2% at grocery stores, 1% on everything else. A strong fit for households with a long commute or frequent road trips."},
    {"name": "Balance Transfer Card", "category": "Low Rate", "annual_fee": 0, "apr_range": "16.99%-24.99%",
     "description": "0% introductory APR on balance transfers for 21 months with a low balance transfer fee, aimed at customers consolidating higher-rate credit card debt from other issuers."},
    {"name": "Small Business Card", "category": "Business", "annual_fee": 0, "apr_range": "17.99%-25.99%",
     "description": "No annual fee business card with 1.5% flat cash back on every purchase, year-end spending summaries for tax season, and no preset spending limit review required."},
    {"name": "Airline Co-Branded Card", "category": "Travel", "annual_fee": 99, "apr_range": "19.99%-26.99%",
     "description": "Earn a free checked bag, priority boarding, and 2x miles on our partner airline, plus a companion fare certificate each account anniversary. Best for customers loyal to one specific airline."},
    {"name": "Hotel Co-Branded Card", "category": "Travel", "annual_fee": 95, "apr_range": "19.99%-26.99%",
     "description": "Automatic elite status with our partner hotel chain, a free night certificate each year, and 5x points on stays at that chain. Best for customers who consistently book the same hotel brand."},
    {"name": "Cashback Flat-Rate Card", "category": "Cash Back", "annual_fee": 0, "apr_range": "18.99%-27.99%",
     "description": "The simplest rewards card we offer: 2% cash back on absolutely everything, no categories, no caps, no annual fee. Built for customers who don't want to think about which card earns more where."},
    {"name": "Premium Metal Card", "category": "Premium", "annual_fee": 695, "apr_range": "19.99%-26.99%",
     "description": "Our most exclusive card: a metal card body, dedicated concierge service, the highest travel credit we offer, and access to invitation-only events. Intended for high-spending customers who value service and status over any single rewards category."},
]

FAQ_CATALOG = [
    ("Disputes", "How do I dispute a charge on my statement?",
     "If you don't recognize a charge or believe it's incorrect, you can open a dispute from the transaction detail screen in the mobile app or by calling the number on the back of your card. Most disputes are resolved within 10 business days, and the disputed amount is temporarily credited to your account while we investigate."),
    ("Disputes", "How long does a dispute investigation take?",
     "Under federal regulations, we have up to 90 days to complete a dispute investigation for most transactions, though the large majority resolve within 10 business days. You'll receive written notice of the outcome either way."),
    ("Disputes", "What happens if I lose a dispute?",
     "If the investigation finds the charge was valid, the temporary credit is reversed and the charge is added back to your balance. You'll receive an explanation of the findings and can request the documentation we used to reach that decision."),
    ("Fraud", "How do I report a lost or stolen card?",
     "Report a lost or stolen card immediately through the mobile app's 'Lock Card' feature, which blocks new transactions instantly, or by calling our 24/7 fraud line. We'll issue a replacement card with a new number, typically arriving within 3-5 business days."),
    ("Fraud", "Am I liable for fraudulent charges on my account?",
     "You are not liable for fraudulent charges reported promptly, consistent with federal law. We monitor for unusual activity automatically and will text or call you if a transaction looks out of pattern."),
    ("Fraud", "What should I do if I suspect identity theft?",
     "Freeze your card in the app immediately, then contact our fraud team so we can review recent activity and, if needed, close and reissue the account with a new number. We also recommend placing a fraud alert with the major credit bureaus."),
    ("Statements", "When is my statement generated each month?",
     "Your statement closes on the same day each month, shown as your 'statement date' in account settings. You can view current-cycle activity at any time in the app even before the statement closes."),
    ("Statements", "Why does my statement balance differ from my current balance?",
     "Your statement balance is a snapshot as of your last statement closing date. Your current balance includes any purchases or payments made since then, which is why the two numbers rarely match exactly."),
    ("Statements", "Can I get paper statements mailed to me?",
     "Yes, paper statements can be enabled in account settings, though there may be a small monthly fee depending on your card product. Electronic statements are always free and available for at least 7 years in the app."),
    ("Fees", "What is a foreign transaction fee and which cards charge it?",
     "A foreign transaction fee is typically 2-3% of the purchase amount charged when you use your card for a transaction processed in a foreign currency. Our Travel Elite Card and Premium Travel Card waive this fee entirely; most other cards in our lineup charge 3%."),
    ("Fees", "What triggers a late payment fee?",
     "A late fee applies if we don't receive at least the minimum payment by the due date shown on your statement. Setting up autopay for at least the minimum payment is the most reliable way to avoid this."),
    ("Fees", "Is there a fee for going over my credit limit?",
     "We do not charge an over-limit fee. Transactions that would exceed your credit limit may simply be declined unless you've opted in to over-limit coverage for a specific card product."),
    ("Credit Limit", "How can I request a credit limit increase?",
     "You can request a credit limit increase in the app under Card Settings, which triggers a soft credit check that does not affect your credit score. Some requests are approved instantly; others may require additional income verification."),
    ("Credit Limit", "Will requesting a credit limit increase hurt my credit score?",
     "Most credit limit increase requests use a soft inquiry, which does not affect your score. In rare cases involving a significant increase, we may perform a hard inquiry, and we'll always disclose that before proceeding."),
    ("Credit Limit", "Why was my credit limit decreased?",
     "Credit limits are periodically reviewed based on payment history, reported income, and overall credit utilization across all your accounts. If your limit is reduced, we send a notice explaining the primary factors that led to the decision."),
    ("Rewards", "How do cash back rewards get paid out?",
     "Cash back accrues automatically as you spend and can be redeemed at any time as a statement credit, direct deposit, or gift card, depending on your card product. There's no minimum redemption amount on most cards."),
    ("Rewards", "Do rewards points expire?",
     "Points earned on our travel cards do not expire as long as your account remains open and in good standing. Cash back on everyday cards also does not expire under the same condition."),
    ("Rewards", "Can I transfer points between accounts?",
     "Points can be transferred to another cardholder on the same account type, or to a partner airline or hotel program at the redemption rates shown in the rewards portal."),
    ("Payments", "What happens if I miss a payment?",
     "A missed payment may result in a late fee and can affect your credit score if it's more than 30 days past due. We recommend contacting us before a due date if you expect to have trouble paying, since hardship options may be available."),
    ("Payments", "Can I change my payment due date?",
     "Yes, you can request a new due date once every 12 months in account settings, subject to a brief processing window before the new date takes effect."),
    ("Payments", "How do autopay minimum vs. full balance options work?",
     "Autopay can be set to pay the minimum due, the full statement balance, or a fixed amount you choose, withdrawn automatically a few days before your due date. You can change or cancel autopay at any time before the withdrawal is scheduled."),
    ("Account Management", "How do I close my credit card account?",
     "You can request account closure through the app or by phone once your balance is paid in full. Closing an account may affect your credit utilization ratio and the average age of your accounts, both of which factor into your credit score."),
    ("Account Management", "Can I add an authorized user to my account?",
     "Yes, authorized users can be added in account settings at no cost on most card products. Authorized users get their own card but are not responsible for the account's payment obligations."),
    ("Account Management", "How do I update my mailing address or phone number?",
     "Contact information can be updated directly in the app under Profile Settings, and changes take effect immediately for statements and fraud alerts."),
    ("Security", "How does the mobile app protect my account?",
     "The app supports biometric login, real-time purchase alerts, and one-tap card freezing. We also use behavioral fraud monitoring that can flag and hold suspicious transactions before they post."),
    ("Security", "Is it safe to use my card for online purchases?",
     "Yes — you can generate a virtual card number for online purchases in the app, which limits exposure if a merchant's systems are ever compromised, since the virtual number can be turned off independently of your physical card."),
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


def generate_card_products():
    products = []
    for i, p in enumerate(CARD_PRODUCT_CATALOG[:CARD_PRODUCTS]):
        products.append({
            "product_id": f"P{i:03d}",
            "name": p["name"],
            "category": p["category"],
            "annual_fee": p["annual_fee"],
            "apr_range": p["apr_range"],
            "description": p["description"],
        })
    return products


def generate_cards(customers, products):
    cards = []
    card_seq = 0
    for cust in customers:
        n_cards = rng.randint(*CARDS_PER_CUSTOMER_RANGE)
        for _ in range(n_cards):
            product = rng_pick(products)
            cards.append({
                "card_id": f"CARD{card_seq:07d}",
                "customer_id": cust["customer_id"],
                "product_id": product["product_id"],
                "opened_date": rand_date(1200, 10),
                "credit_limit": rng.choice([1000, 2000, 5000, 7500, 10000, 15000, 20000, 30000]),
                "current_balance": round(rng.uniform(0, 8000), 2),
            })
            card_seq += 1
    return cards


def generate_transactions(cards):
    transactions = []
    for i in range(TRANSACTIONS):
        card = rng_pick(cards)
        category = rng_pick(CATEGORY_NAMES)
        cat = CATEGORIES[category]
        merchant = rng_pick(cat["merchants"])
        template = rng_pick(cat["templates"])
        description = template.format(m=merchant)
        amount = round(abs(rng.gauss(45, 60)) + 3, 2)
        transactions.append({
            "transaction_id": f"T{i:08d}",
            "card_id": card["card_id"],
            "customer_id": card["customer_id"],
            "merchant": merchant,
            "category": category,
            "amount": amount,
            "date": rand_date(180, 0),
            "description": description,
        })
    return transactions


def generate_faq_articles():
    articles = []
    for i, (category, title, body) in enumerate(FAQ_CATALOG[:FAQ_ARTICLES]):
        articles.append({
            "article_id": f"FAQ{i:03d}",
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

    print("Generating card products...")
    products = generate_card_products()
    write_jsonl(DATA_FILES["card_products"], products)
    print(f"  {len(products):,} card products")

    print("Generating cards...")
    cards = generate_cards(customers, products)
    write_jsonl(DATA_FILES["cards"], cards)
    print(f"  {len(cards):,} cards")

    print("Generating transactions...")
    transactions = generate_transactions(cards)
    write_jsonl(DATA_FILES["transactions"], transactions)
    print(f"  {len(transactions):,} transactions")

    print("Generating FAQ articles...")
    faqs = generate_faq_articles()
    write_jsonl(DATA_FILES["faq_articles"], faqs)
    print(f"  {len(faqs):,} FAQ articles")


if __name__ == "__main__":
    main()
