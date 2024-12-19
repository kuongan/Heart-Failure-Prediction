from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from typing import List
import shutil
import os
import pickle
import numpy as np
from fastapi import FastAPI, Form
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import io
from joblib import load
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, auc, roc_curve, accuracy_score
app = FastAPI()

# Dummy data paths
DATA_FOLDER = "/home/nguyen-minh-hieu/data_mining/Heart-Failure-Prediction/data"
MODEL_FOLDER = "/home/nguyen-minh-hieu/data_mining/Heart-Failure-Prediction/models"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
DEVICE = "cpu"  # Thiết bị cho Tensor
SEED = 42  # Random seed
params=[1, 25,0.2678837376135781, 0.00793912222669994]

n_layers = params[0]

units_1 = 19
dropout_1 = np.round(params[2],5)
def get_scores(y, y_pred):
    data={'Accuracy': np.round(accuracy_score(y, y_pred),2),
    'Precision':np.round(precision_score(y, y_pred),2),
    'Recall':np.round(recall_score(y, y_pred),2),
    'F1':np.round(f1_score(y, y_pred),2),
    'ROC AUC':np.round(roc_auc_score(y, y_pred),2)}
    scores_df = pd.Series(data).to_frame('scores')
    return scores_df

def conf_matrix(y, y_pred):
    fig, ax =plt.subplots(figsize=(3.5,3.5))
    labels=['No','Yes']
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size":25})
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
def plot_roc_curve(y_test, y_pred_prob_list):
    # Vẽ ROC curve và lưu thành ảnh
    plt.figure(figsize=(5.5, 4))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_list)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
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

lr = np.round(params[3],8)
# Tạo dataset cho PyTorch
class CustomDataset:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(15, units_1)
        self.layer_out = nn.Linear(units_1, 1) 
        self.dropout1 = nn.Dropout(p=dropout_1)
        
    def forward(self, inputs):
        x = F.relu(self.layer_1(inputs))
        x = self.dropout1(x)
        x = self.layer_out(x)
        
        return x
#xu ly du lieu

df = pd.read_csv('/home/nguyen-minh-hieu/data_mining/Heart-Failure-Prediction/data/heart.csv', skipinitialspace=True)
df_ml=df.copy()
df_ml = pd.get_dummies(df_ml, drop_first=True)
X = df_ml.drop(["HeartDisease"], axis=1)
y = df_ml["HeartDisease"]
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X, y, test_size=0.15, stratify = y, random_state = 42)
# Tiền xử lý dữ liệu
df = df[df['RestingBP'] > 0]
df['Cholesterol'] = df['Cholesterol'].replace({0:np.nan})
df['Sex'] = df['Sex'].replace({'M': 0, 'F': 1})
df['ExerciseAngina'] = df['ExerciseAngina'].replace({'N': 0, 'Y': 1})


# One-hot encoding
encoded_df = pd.get_dummies(df, drop_first=True)
# print(encoded_df['ST_Slope'])
# Chia dữ liệu
X = encoded_df.drop('HeartDisease', axis=1)
y = encoded_df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
chol = 240

X_train['Cholesterol'] = X_train['Cholesterol'].fillna(chol)

X_test['Cholesterol'] = X_test['Cholesterol'].fillna(chol)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_scaled_ml = scaler.fit_transform(X_train_ml)
X_test_scaled_ml = scaler.transform(X_test_ml)
# Áp dụng PCA
pca = PCA()
pca.fit_transform(X_train);

# Chuyển dữ liệu sang PyTorch tensors
test_data = CustomDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test.values))
test_loader = DataLoader(dataset=test_data, batch_size=1)
# 1. **Data Upload**
# @app.post("/upload/")
# async def upload_data(file: UploadFile = File(...)):
#     file_location = os.path.join(DATA_FOLDER, file.filename)
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return {"message": f"File '{file.filename}' uploaded successfully!"}

# 2. **Train Model**
# Endpoint: Predict sử dụng một mô hình được chọn
@app.post("/predict_DL/")
async def predict_model(model_name: str = Form(...)):
    """
    Predict test data using a specific model selected by the user.
    Args:
        model_name (str): Tên mô hình do người dùng cung cấp.
    Returns:
        dict: Kết quả prediction và các metrics.
    """
    # Tạo dataframe giả lập (cần thay bằng dữ liệu thực tế)

    # Load mô hình
    # Load mô hình từ checkpoint PyTorch Lightning
    if model_name== 'DL':
        # model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pt")
        # print("Loading model from:", model_path)

        # if not os.path.exists(model_path):
        #     return {"error": f"Checkpoint '{model_name}' not found!"}



    # Khởi tạo mô hình và load weights
        model = Net()

        model.load_state_dict(torch.load(r'C:\Edisk\tailieusinhvien\hk5\data_mining\backend\models\state_dict.pt'))
        
        # Predict trên tập test
        y_pred_prob_list = []
        prediction_list = []


        # Duyệt qua từng batch trong test_loader
        with torch.no_grad():  # Vô hiệu hóa tính toán gradient
            model.eval()
            for X_test_batch, y_test_batch in test_loader:
                # Dự đoán logits từ mô hình
                X_test_batch = X_test_batch.to(DEVICE)
                #PREDICTION
                output = model(X_test_batch)
                y_pred_prob = torch.sigmoid(output)
                y_pred_prob_list.append(y_pred_prob.cpu().numpy())
                y_pred = torch.round(y_pred_prob)
                prediction_list.append(y_pred.cpu().numpy())

        # Chuyển kết quả thành list (nếu cần)
        y_pred_prob_list = [a.squeeze().tolist() for a in y_pred_prob_list]
        prediction_list = [a.squeeze().tolist() for a in prediction_list]
            # Trả kết quả
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(y_pred_prob_list), torch.tensor(y_test.values).float()
        ).item()

        # Tính accuracy
        accuracy = accuracy_score(y_test, prediction_list)

        # Tính confusion matrix
        confusion = confusion_matrix(y_test, prediction_list).tolist()

        # Tính classification report
        report = classification_report(y_test, prediction_list, output_dict=True)
        print(classification_report(y_test, prediction_list))

        """## Confusion Matrix"""
        roc_buf = plot_roc_curve(y_test, y_pred_prob_list)

        # Tạo Confusion Matrix
        conf_buf = conf_matrix(y_test, prediction_list)
        roc_path = os.path.join(DATA_FOLDER, "roc_curve_DL.png")
        conf_path = os.path.join(DATA_FOLDER, "confusion_matrix_DL.png")
        # Trả kết quả
        with open(roc_path, "wb") as f:
            f.write(roc_buf.getvalue())
        with open(conf_path, "wb") as f:
            f.write(conf_buf.getvalue())
        return {
            "roc_curve": roc_path,
            "confusion_matrix": conf_path,
            "classification_report": report
        }
    if model_name=="XGB":
        #xgb_model_path = os.path.join(MODEL_FOLDER, "xgb_grid_model.pkl")
        
        #if not os.path.exists(xgb_model_path):
         #   return {"error": f"Model file '{xgb_model_path}' not found!"}

        # Load mô hình từ file pickle
        xgb_model = load("/home/nguyen-minh-hieu/data_mining/Heart-Failure-Prediction/models/xgb_grid_model.pkl")

        # Dự đoán xác suất và nhãn
        y_pred_prob_list = xgb_model.predict_proba(X_test_ml)[:, 1]  # Lấy xác suất lớp 1
        prediction_list = xgb_model.predict(X_test_ml)

        # Tính toán các chỉ số đánh giá
      
        accuracy = accuracy_score(y_test_ml, prediction_list)
        confusion = confusion_matrix(y_test_ml, prediction_list).tolist()
        report = classification_report(y_test_ml, prediction_list, output_dict=True)

        # Vẽ ROC và Confusion Matrix
        roc_buf = plot_roc_curve(y_test_ml, y_pred_prob_list)
        conf_buf = conf_matrix(y_test_ml, prediction_list)

        roc_path = os.path.join(DATA_FOLDER, "roc_curve_XGB.png")
        conf_path = os.path.join(DATA_FOLDER, "confusion_matrix_XGB.png")

        with open(roc_path, "wb") as f:
            f.write(roc_buf.getvalue())
        with open(conf_path, "wb") as f:
            f.write(conf_buf.getvalue())

        return {
            "roc_curve": roc_path,
            "confusion_matrix": conf_path,
            "classification_report": report
        }
# 3. **Prediction**
# class PredictionInput(BaseModel):
#     features: List[float]

# @app.post("/predict1/")
# async def predict(data: PredictionInput, model_name: str):
#     model_path = os.path.join(MODEL_FOLDER, f"{model_name}.ckpt")
#     print(model_path)
#     if not os.path.exists(model_path):
#         return {"error": f"Model '{model_name}' not found!"}

#     # Load the trained model
#     with open(model_path, "rb") as f:
#         model = pickle.load(f)
    
#     # Convert input data
#     input_data = np.array(data.features).reshape(1, -1)
#     prediction = model.predict(input_data)
#     return {"prediction": prediction.tolist()}