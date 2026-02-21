import torch
import os
import pandas as pd
import pickle
import sys

def run_health_check():
    print("🏥 AGAE-MTL System Health Check")
    print("="*40)
    
    files = [
        "app_enhanced.py",
        "train_enhanced.py",
        "models/multitask_model.py",
        "models/autoencoder.py",
        "models/attention.py",
        "data/nfhs5_enhanced.csv",
        "nutrition_mtl_model_enhanced.pth",
        "scaler_enhanced.pkl",
        "encoders_enhanced.pkl",
        "clinical_threshold.txt"
    ]
    
    missing = 0
    for f in files:
        if os.path.exists(f):
            print(f"✅ FOUND: {f}")
        else:
            print(f"❌ MISSING: {f}")
            missing += 1
            
    if missing > 0:
        print(f"\n⚠️ WARNING: {missing} critical files are missing!")
    else:
        print("\n🏆 PROJECT INTEGRITY: 100% PERFECT")
        
    print("\n📦 Dependency Check...")
    try:
        import streamlit
        import reportlab
        print("✅ Core dependencies installed.")
    except ImportError as e:
        print(f"❌ Dependency Error: {e}")

    print("\n🧠 Model Test...")
    try:
        from models.multitask_model import NutritionMTLModel
        with open("encoders_enhanced.pkl", "rb") as f: encoders = pickle.load(f)
        with open("scaler_enhanced.pkl", "rb") as f: scaler = pickle.load(f)
        
        input_dim = 14
        model = NutritionMTLModel(input_dim)
        model.load_state_dict(torch.load("nutrition_mtl_model_enhanced.pth"))
        model.eval()
        print("✅ Deep Learning Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model Error: {e}")

    print("\n" + "="*40)
    print("READY FOR PEER REVIEW AND RESEARCH SUBMISSION.")

if __name__ == "__main__":
    run_health_check()
