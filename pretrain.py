"""
Pre-training CLI script for TrafficFlow AI.
Prepares datasets, seeds database, and trains the initial LSTM model.
"""
import sys
import os

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import init_db
from src.auth import seed_default_users
from src.data_pipeline import clean_and_engineer_features
from src.model_trainer import train_traffic_model

def main():
    print("==================================================")
    print(" TrafficFlow AI - Pre-training & Setup Pipeline")
    print("==================================================")
    
    print("\n[1/4] Initializing SQLite database...")
    init_db()
    print("Database initialized.")

    print("\n[2/4] Seeding default demo accounts...")
    seed_default_users()
    print("Default accounts ready:")
    print("  - Admin:     username='admin',        pwd='Admin@Traffic2026'")
    print("  - Analyst:   username='analyst',      pwd='Analyst@Traffic2026'")
    print("  - Executive: username='mba_executive',pwd='Executive@Traffic2026'")

    print("\n[3/4] Validating and cleaning traffic dataset...")
    df = clean_and_engineer_features(save_to_disk=True)
    print(f"Processed {len(df)} records.")

    print("\n[4/4] Training LSTM Neural Network (8 epochs)...")
    result = train_traffic_model(
        epochs=8,
        batch_size=64,
        sequence_length=12,
        learning_rate=0.001,
        username="admin"
    )

    metrics = result["test_metrics"]
    print("\n==================================================")
    print(" Training Successful! Holdout Test Set Evaluation:")
    print(f"  - R² Score:              {metrics['r2']:.4f}")
    print(f"  - RMSE:                  {metrics['rmse']:.2f} vehicles/hour")
    print(f"  - MAE:                   {metrics['mae']:.2f} vehicles/hour")
    print(f"  - MAPE:                  {metrics['mape']:.2f}%")
    print(f"  - Explained Variance:    {metrics['explained_variance']:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
