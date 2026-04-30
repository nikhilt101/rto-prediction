import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
N = 10000

# --- Feature definitions ---
payment_modes = ['COD', 'Prepaid', 'UPI', 'Card']
product_categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Books', 'Toys', 'Sports']
seller_tiers = ['Gold', 'Silver', 'Bronze']
zones = ['Metro', 'Tier1', 'Tier2', 'Rural']
time_slots = ['Morning', 'Afternoon', 'Evening']

payment = np.random.choice(payment_modes, N, p=[0.45, 0.30, 0.15, 0.10])
category = np.random.choice(product_categories, N)
seller_tier = np.random.choice(seller_tiers, N, p=[0.20, 0.45, 0.35])
zone = np.random.choice(zones, N, p=[0.30, 0.25, 0.30, 0.15])
time_slot = np.random.choice(time_slots, N)

order_value = np.where(
    payment == 'COD',
    np.random.lognormal(7.0, 0.6, N),
    np.random.lognormal(7.5, 0.5, N)
).clip(100, 50000)

delivery_attempts = np.random.choice([1, 2, 3], N, p=[0.55, 0.30, 0.15])
customer_rating = np.random.choice([1, 2, 3, 4, 5], N, p=[0.08, 0.10, 0.20, 0.35, 0.27])
days_to_deliver = np.random.choice(range(1, 11), N, p=[0.05,0.10,0.20,0.25,0.18,0.10,0.05,0.04,0.02,0.01])
address_quality_score = np.random.uniform(0.2, 1.0, N).round(2)
is_repeat_customer = np.random.choice([0, 1], N, p=[0.40, 0.60])
holiday_season = np.random.choice([0, 1], N, p=[0.75, 0.25])
seller_rating = np.random.uniform(2.5, 5.0, N).round(1)
pincode_density = np.random.choice(['High', 'Medium', 'Low'], N, p=[0.40, 0.35, 0.25])

# --- RTO probability (realistic domain logic) ---
rto_prob = np.zeros(N)

# COD has highest RTO
rto_prob += np.where(payment == 'COD', 0.20, 0.03)

# Zone effect
zone_effect = {'Metro': 0.00, 'Tier1': 0.04, 'Tier2': 0.08, 'Rural': 0.16}
rto_prob += np.array([zone_effect[z] for z in zone])

# Delivery attempts
rto_prob += (delivery_attempts - 1) * 0.08

# Address quality
rto_prob += (1 - address_quality_score) * 0.15

# Low customer rating = higher RTO
rto_prob += np.where(customer_rating <= 2, 0.08, 0.0)

# New customers are riskier
rto_prob += np.where(is_repeat_customer == 0, 0.05, 0.0)

# Days to deliver
rto_prob += np.where(days_to_deliver > 7, 0.07, 0.0)

# High order value + COD = risky
rto_prob += np.where((order_value > 5000) & (payment == 'COD'), 0.06, 0.0)

# Seller quality
seller_effect = {'Gold': -0.03, 'Silver': 0.0, 'Bronze': 0.05}
rto_prob += np.array([seller_effect[s] for s in seller_tier])

# Pincode density
density_effect = {'High': -0.02, 'Medium': 0.02, 'Low': 0.08}
rto_prob += np.array([density_effect[p] for p in pincode_density])

# Holiday season
rto_prob += np.where(holiday_season == 1, 0.03, 0.0)

rto_prob = np.clip(rto_prob, 0.02, 0.92)
rto = (np.random.uniform(0, 1, N) < rto_prob).astype(int)

df = pd.DataFrame({
    'payment_mode': payment,
    'product_category': category,
    'seller_tier': seller_tier,
    'delivery_zone': zone,
    'time_slot': time_slot,
    'order_value': order_value.round(2),
    'delivery_attempts': delivery_attempts,
    'customer_rating': customer_rating,
    'days_to_deliver': days_to_deliver,
    'address_quality_score': address_quality_score,
    'is_repeat_customer': is_repeat_customer,
    'holiday_season': holiday_season,
    'seller_rating': seller_rating,
    'pincode_density': pincode_density,
    'rto': rto
})

print(f"Dataset shape: {df.shape}")
print(f"RTO rate: {df['rto'].mean():.3f} ({df['rto'].sum()} returns out of {N})")
print(f"\nRTO by payment mode:\n{df.groupby('payment_mode')['rto'].mean().round(3)}")
print(f"\nRTO by zone:\n{df.groupby('delivery_zone')['rto'].mean().round(3)}")

df.to_csv('/home/claude/rto_research/rto_dataset.csv', index=False)
print("\n✅ Dataset saved.")
