import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate mock customer data
num_customers = 500

customer_ids = [f"{i:04d}-XYZ" for i in range(1, num_customers + 1)]
genders = np.random.choice(["Male", "Female"], size=num_customers)
senior_citizens = np.random.choice([0, 1], p=[0.85, 0.15], size=num_customers)
partners = np.random.choice(["Yes", "No"], size=num_customers)
dependents = np.random.choice(["Yes", "No"], p=[0.3, 0.7], size=num_customers)
tenures = np.random.randint(1, 72, size=num_customers)
contracts = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.5, 0.25, 0.25], size=num_customers)
paperless = np.random.choice(["Yes", "No"], p=[0.6, 0.4], size=num_customers)
payment_methods = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=num_customers
)

# Charges generation
monthly_charges = np.random.uniform(20.0, 118.0, size=num_customers)
# Total charges is roughly tenure * monthly_charges + some noise
total_charges = tenures * monthly_charges + np.random.normal(0, 50, size=num_customers)
total_charges = np.clip(total_charges, 20.0, None)  # Ensure non-negative/reasonable min

# Define probability of churn based on parameters to make model learning interesting
# Higher churn if: short tenure, month-to-month contract, high monthly charges, electronic check payment
churn_prob = []
for i in range(num_customers):
    prob = 0.1  # base probability
    if tenures[i] < 12:
        prob += 0.35
    elif tenures[i] < 24:
        prob += 0.15
        
    if contracts[i] == "Month-to-month":
        prob += 0.25
        
    if monthly_charges[i] > 80.0:
        prob += 0.15
        
    if payment_methods[i] == "Electronic check":
        prob += 0.1
        
    # Cap between 0.05 and 0.95
    prob = np.clip(prob, 0.05, 0.95)
    churn_prob.append(prob)

churns = [("Yes" if np.random.random() < p else "No") for p in churn_prob]

# Create DataFrame
df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": genders,
    "SeniorCitizen": senior_citizens,
    "Partner": partners,
    "Dependents": dependents,
    "tenure": tenures,
    "Contract": contracts,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_methods,
    "MonthlyCharges": np.round(monthly_charges, 2),
    "TotalCharges": np.round(total_charges, 2),
    "churn": churns
})

# Save to uploads folder
output_path = os.path.join(os.path.dirname(__file__), "uploads", "churn_customer_data.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"SUCCESS: Generated mock churn dataset with {num_customers} rows.")
print(f"Saved to: {output_path}")
print("Churn distribution:")
print(df["churn"].value_counts())
