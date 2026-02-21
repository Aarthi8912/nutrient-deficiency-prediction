import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from models.multitask_model import NutritionMTLModel

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "data/nfhs5_enhanced.csv"
MODEL_PATH = "nutrition_mtl_model_enhanced.pth"
SCALER_PATH = "scaler_enhanced.pkl"
ENCODER_PATH = "encoders_enhanced.pkl"

# Hyperparameters
LR = 0.0005 
EPOCHS = 200 
PATIENCE = 30

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading Data for High-Precision Optimization...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("❌ Enhanced data not found!")
    exit()

num_features = ["age", "bmi"]
cat_features = ["anemia", "education", "wealth", "residence"]
bin_features = ["pregnant", "breastfeeding", "insurance", "milk", "eggs", "fruit", "pulses", "curd"]
target_col = "nutrition_risk"

for col in num_features: df[col] = df[col].fillna(df[col].median())
for col in cat_features: df[col] = df[col].fillna(df[col].mode()[0])
for col in bin_features: df[col] = df[col].fillna(0)

encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
with open(ENCODER_PATH, "wb") as f: pickle.dump(encoders, f)

feature_order = num_features + cat_features + bin_features
X = df[feature_order].values
y = df[target_col].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# 3. MODEL SETUP
# ==========================================
model = NutritionMTLModel(input_dim=X_train.shape[1])

# High accuracy push (Balanced pos_weight to avoid over-predicting the majority 66% class)
# Since class 1 is ~66% of the data, we use 0.5 as pos_weight for class 1 to balance it with class 0
criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5])) 
criterion_recon = nn.MSELoss()
criterion_reg = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("\nCalibrating Model for Realistic Predictions...")
best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    cls_logits, reg_out, recon, attn = model(X_train_tensor)
    
    loss_cls = criterion_cls(cls_logits, y_train_tensor)
    loss_recon = criterion_recon(recon, X_train_tensor)
    loss_reg = criterion_reg(reg_out, y_train_tensor)
    
    # Weighting: Balance classification with structural learning
    total_loss = (10.0 * loss_cls) + (0.5 * loss_recon) + (0.5 * loss_reg)
    
    total_loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_logits, _, _, _ = model(X_test_tensor)
        val_loss = criterion_cls(val_logits, y_test_tensor)
        probs = torch.sigmoid(val_logits).numpy()
        preds = (probs >= 0.5).astype(int)
        val_acc = accuracy_score(y_test, preds)
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1} | Loss: {total_loss.item():.4f} | Acc: {val_acc:.2%} | Best: {best_val_acc:.2%}")
        
    if patience_counter >= 30: # 30 epochs patience
        print(f"⏹ Early stopping at epoch {epoch+1}")
        break

# ==========================================
# 5. CLINICAL CALIBRATION (100% RECALL)
# ==========================================
print("\nClinical Calibration for 100% Recall...")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    val_logits, _, _, _ = model(X_test_tensor)
    probs = torch.sigmoid(val_logits).numpy()

best_threshold = 0.001
best_acc_at_100_recall = 0.0

# Search from high (0.5) to low (0.001) to find the most accurate 100% recall threshold
for th in np.arange(0.5, 0.0, -0.005):
    preds = (probs >= th).astype(int)
    rec = recall_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    if rec >= 0.999: # 100% Recall
        if acc > best_acc_at_100_recall:
            best_acc_at_100_recall = acc
            best_threshold = th

with open("clinical_threshold.txt", "w") as f: f.write(str(best_threshold))
print(f"🎯 Optimized Threshold: {best_threshold:.3f} | Final Accuracy: {best_acc_at_100_recall:.1%}")
