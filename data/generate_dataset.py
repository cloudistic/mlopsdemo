"""
Enterprise Dataset Generator: Customer Churn & Lifetime Value (5,000 Records)
Generates realistic tabular data with 18 features, class imbalance, and realistic noise
for evaluating models in MLflow, Airflow, and H2O.
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def get_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic customer churn dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.70, 0.30],  # 70% retained, 30% churned
        flip_y=0.03,            # 3% label noise
        class_sep=0.85,
        random_state=random_state
    )

    feature_names = [
        'tenure_months', 'monthly_charges', 'total_charges', 'age', 'credit_score',
        'support_tickets', 'login_frequency', 'avg_session_duration', 'payment_delay_days',
        'contract_length_months', 'num_dependents', 'international_plan', 'voice_mail_plan',
        'data_usage_gb', 'roaming_charges', 'device_protection', 'paperless_billing', 'num_referrals'
    ]

    df = pd.DataFrame(X, columns=feature_names)
    
    # Scale/transform to realistic enterprise ranges
    df['tenure_months'] = np.clip(np.round((df['tenure_months'] + 3) * 10), 1, 72)
    df['monthly_charges'] = np.clip(np.round((df['monthly_charges'] + 3) * 20 + 20, 2), 19.99, 149.99)
    df['total_charges'] = np.round(df['tenure_months'] * df['monthly_charges'] * np.random.uniform(0.9, 1.1, len(df)), 2)
    df['age'] = np.clip(np.round((df['age'] + 3) * 8 + 18), 18, 80).astype(int)
    df['credit_score'] = np.clip(np.round((df['credit_score'] + 3) * 60 + 500), 350, 850).astype(int)
    df['support_tickets'] = np.clip(np.round(np.abs(df['support_tickets']) * 2), 0, 10).astype(int)
    df['payment_delay_days'] = np.clip(np.round(np.abs(df['payment_delay_days']) * 5), 0, 30).astype(int)
    df['num_dependents'] = np.clip(np.round(np.abs(df['num_dependents'])), 0, 5).astype(int)
    df['international_plan'] = (df['international_plan'] > 0).astype(int)
    df['voice_mail_plan'] = (df['voice_mail_plan'] > 0).astype(int)
    df['device_protection'] = (df['device_protection'] > 0).astype(int)
    df['paperless_billing'] = (df['paperless_billing'] > 0).astype(int)
    df['num_referrals'] = np.clip(np.round(np.abs(df['num_referrals']) * 2), 0, 10).astype(int)

    df['churn_target'] = y
    return df


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "customer_churn_5k.csv")
    
    df = get_churn_dataset(5000)
    df.to_csv(out_csv, index=False)
    print(f"✓ Generated realistic churn dataset: {out_csv} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print("Class breakdown:\n", df['churn_target'].value_counts())
