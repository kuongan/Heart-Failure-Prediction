from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import List
import os
import io
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score,
    auc, roc_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from joblib import load, dump
from sklearn.decomposition import PCA

# Initialize FastAPI app
app = FastAPI()

# Configuration
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    "n_layers": 1,
    "units_1": 14,
    "dropout_1": 0.57151,
    "learning_rate": 0.00639
}

# Helper Functions
def get_scores(y, y_pred):
    """Calculate various model performance metrics."""
    data = {
        'Accuracy': np.round(accuracy_score(y, y_pred), 2),
        'Precision': np.round(precision_score(y, y_pred), 2),
        'Recall': np.round(recall_score(y, y_pred), 2),
        'F1': np.round(f1_score(y, y_pred), 2),
        'ROC AUC': np.round(roc_auc_score(y, y_pred), 2)
    }
    return pd.Series(data).to_frame('scores')

def create_confusion_matrix(y, y_pred):
    """Generate confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    labels = ['No', 'Yes']
    sns.heatmap(
        confusion_matrix(y, y_pred),
        annot=True,
        cmap="Blues",
        fmt='g',
        cbar=False,
        annot_kws={"size": 25}
    )
    plt.title('Heart Failure?', fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17)
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test')
    ax.set_xlabel('Predicted')
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

def create_roc_curve(y_test, y_pred_prob):
    """Generate ROC curve plot."""
    plt.figure(figsize=(5.5, 4))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.legend(loc='lower right', fontsize=12, fancybox=True, shadow=True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_size, MODEL_PARAMS["units_1"])
        self.layer_out = nn.Linear(MODEL_PARAMS["units_1"], 1)
        self.dropout1 = nn.Dropout(p=MODEL_PARAMS["dropout_1"])
    
    def forward(self, inputs):
        x = F.relu(self.layer_1(inputs))
        x = self.dropout1(x)
        return self.layer_out(x)

# Data Processing
def prepare_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'), skipinitialspace=True)
    
    # Data cleaning
    df = df[df['RestingBP'] > 0]
    df['Cholesterol'] = df['Cholesterol'].replace({0: np.nan})
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
    
    # One-hot encoding
    encoded_df = pd.get_dummies(df, drop_first=True)
    
    # Split features and target
    X = encoded_df.drop('HeartDisease', axis=1)
    y = encoded_df['HeartDisease']
    
    # Train-test split
    X_train, X_test_DL, y_train, y_test_DL = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Handle missing values
    X_train['Cholesterol'].fillna(240, inplace=True)
    X_test['Cholesterol'].fillna(240, inplace=True)
    X_test_DL['Cholesterol'].fillna(240, inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    X_test_DL = pd.DataFrame(
        scaler.transform(X_test_DL),
        columns=X_test_DL.columns,
        index=X_test_DL.index
    )
    
    # Create validation split
    X_train_arr, X_valid, y_train_arr, y_valid = train_test_split(
        X_train.values, y_train.values,
        test_size=0.125,
        stratify=y_train,
        random_state=SEED
    )
    
    # Convert to numpy arrays for PyTorch
    X_test_DL = X_test_DL.values
    y_test_DL = y_test_DL.values
    
    return X_train_arr, X_valid, X_test, y_train_arr, y_valid, y_test, X_test_DL, y_test_DL



# API Endpoints
@app.post("/modeling/")
async def predict_model(model_name: str = Form(...)):
    """
    Train and evaluate a model specified by model_name.
    
    Args:
        model_name: Either 'DL' for Deep Learning or 'XGB' for XGBoost
    
    Returns:
        dict: Model evaluation metrics and plots
    """
    X_train, X_valid, X_test, y_train, y_valid, y_test ,X_test_DL,y_test_DL= prepare_data()
    
    if model_name == 'DL':
        return await train_evaluate_dl(X_train, X_valid, X_test_DL, y_train, y_valid, y_test_DL)
    elif model_name == 'XGB':
        return await train_evaluate_xgb(X_train, X_test, y_train, y_test)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid model_name. Choose 'DL' or 'XGB'"}
        )

async def train_evaluate_dl(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """Train and evaluate deep learning model."""
    # Create data loaders with numpy arrays
    train_loader = DataLoader(
        CustomDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        ),
        batch_size=16
    )
    valid_loader = DataLoader(
        CustomDataset(
            torch.FloatTensor(X_valid),
            torch.FloatTensor(y_valid)
        ),
        batch_size=1
    )
    test_loader = DataLoader(
        CustomDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        ),
        batch_size=1
    )
    
    # Initialize model and training parameters
    model = Net(X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MODEL_PARAMS["learning_rate"],
        weight_decay=0.0001
    )
    
    # Training loop
    model = train_dl_model(
        model, train_loader, valid_loader,
        criterion, optimizer, num_epochs=100
    )
    
    # Evaluate model
    y_pred_prob, y_pred = evaluate_dl_model(model, test_loader)
    
    # Generate evaluation metrics and plots
    return create_evaluation_results(
        y_test, y_pred, y_pred_prob,
        model_suffix='DL'
    )

async def train_evaluate_xgb(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost model."""
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 6],
        "learning_rate": [0.1, 0.3],
        "subsample": [0.5, 1],
        "colsample_bytree": [0.5, 1]
    }
    
    model = GridSearchCV(
        XGBClassifier(random_state=SEED),
        param_grid,
        scoring="f1",
        verbose=2,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    dump(model, os.path.join(MODEL_DIR, "xgb_grid_model.pkl"))
    
    # Predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    return create_evaluation_results(
        y_test, y_pred, y_pred_prob,
        model_suffix='XGB'
    )

def create_evaluation_results(y_test, y_pred, y_pred_prob, model_suffix):
    """Create and save evaluation metrics and plots."""
    # Generate plots
    roc_buf = create_roc_curve(y_test, y_pred_prob)
    conf_buf = create_confusion_matrix(y_test, y_pred)
    
    # Save plots
    roc_path = os.path.join(DATA_DIR, f"roc_curve_{model_suffix}.png")
    conf_path = os.path.join(DATA_DIR, f"confusion_matrix_{model_suffix}.png")
    
    with open(roc_path, "wb") as f:
        f.write(roc_buf.getvalue())
    with open(conf_path, "wb") as f:
        f.write(conf_buf.getvalue())
    
    return {
        "roc_curve": roc_path,
        "confusion_matrix": conf_path,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

def train_dl_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=100):
    """Train deep learning model with early stopping."""
    early_stopping_patience = 15
    early_stopping_counter = 0
    valid_loss_min = np.inf
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        
        valid_loss = validate_epoch(model, valid_loader, criterion)
        
        if valid_loss < valid_loss_min:
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, 'state_dict.pt')
            )
            valid_loss_min = valid_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter > early_stopping_patience:
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'state_dict.pt')))
    return model

def train_epoch(model, train_loader, criterion, optimizer):
    """Train one epoch."""
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_epoch(model, valid_loader, criterion):
    """Validate one epoch."""
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            running_loss += loss.item()
    return running_loss / len(valid_loader)

def evaluate_dl_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    y_pred_prob_list = []
    prediction_list = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            y_pred_prob = torch.sigmoid(output)
            y_pred = torch.round(y_pred_prob)
            
            y_pred_prob_list.append(y_pred_prob.cpu().numpy().squeeze())
            prediction_list.append(y_pred.cpu().numpy().squeeze())
    
    return y_pred_prob_list, prediction_list