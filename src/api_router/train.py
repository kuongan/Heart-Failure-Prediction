import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
import seaborn as sns 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import sys
from src.api_router.preprocess import Preprocessor 
sys.modules["__main__"].Preprocessor = Preprocessor

router = APIRouter()

# Define the parameters schema
class TrainParams(BaseModel):
    model: str
    n_estimators: int = None
    learning_rate: float = None
    max_depth: int = None
    subsample: float = None
    epochs: int = None
    num_layers: int = None
    num_node: int = None
def save_classification_report_as_txt(y_true, y_pred, classification_report_path: str):
    """
    Saves the classification report as a .txt file.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classification_report_path: Path where the .txt file will be saved.
    """
    # Generate the classification report
    class_report = classification_report(y_true, y_pred)
    
    # Save the report as a .txt file
    with open(classification_report_path, "w") as f:
        f.write(class_report)


def save_txt_as_image(txt_file_path: str, image_path: str):
    """
    Converts the .txt file containing the classification report into an image with minimal background.

    Args:
        txt_file_path: Path to the .txt file.
        image_path: Path where the image will be saved.
    """
    # Read the text file
    with open(txt_file_path, "r") as file:
        text = file.read()
    
    # Create a figure and axis to plot the text
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True)
    
    # Remove axis
    ax.axis("off")

    # Adjust figure size based on text
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bbox.width + 0.5, bbox.height + 0.5)  # Add small padding

    # Save the plot as an image with tight bounding box
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Helper function to save confusion matrix
def save_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Helper function to save ROC curve
def save_roc_curve(y_true, y_pred, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_pred):.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Training endpoint
@router.post("/train")
def train_model(params: TrainParams):
    try:
        # Load and preprocess data
        data_path = "data/heart.csv"
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Data file not found")

        df = pd.read_csv(data_path)
        X = df.drop(columns="HeartDisease")
        y = df["HeartDisease"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Define preprocessing
        num_rest_attribs = ["Age", "MaxHR", "Oldpeak"]
        num_missvalue_attribs = ["RestingBP", "Cholesterol"]
        cat_nominal_attribs = ["ChestPainType", "RestingECG", "ST_Slope"]
        cat_binary_attribs = ["Sex", "ExerciseAngina"]

        prep = Preprocessor(
            num_rest_attribs, num_missvalue_attribs, cat_nominal_attribs, cat_binary_attribs
        )
        X_train_preprocessed = prep.fit_transform(X_train)
        X_test_preprocessed = prep.transform(X_test)

        # Train model based on selected type
        if params.model == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                max_depth=params.max_depth,
                subsample=params.subsample,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train_preprocessed, y_train)
            y_pred = model.predict(X_test_preprocessed)
            y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]
            print('done')

        elif params.model == "neural_network":
            model = Sequential()
            model.add(Dense(params.num_node, activation="relu", input_shape=(X_train_preprocessed.shape[1],)))
            for _ in range(params.num_layers - 1):
                model.add(Dense(params.num_node, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))

            model.compile(optimizer=Adam(learning_rate=params.learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X_train_preprocessed, y_train, epochs=params.epochs, batch_size=32, verbose=1)

            y_pred_proba = model.predict(X_test_preprocessed).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

        else:
            raise HTTPException(status_code=400, detail="Unsupported model")

        # Evaluate and save results
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(class_report, conf_matrix, roc_auc)
        # Ensure the output directory exists
        os.makedirs("static/images", exist_ok=True)

        # Define output file paths
        classification_report_path_txt = "src/static/images/generated_classification_report.txt"
        confusion_matrix_path = "src/static/images/generated_confusion_matrix.jpg"
        roc_curve_path = "src/static/images/generated_roc_auc.jpg"
        classification_report_path = "src/static/images/generated_classification_report.jpg"
        # Save the confusion matrix and ROC curve
        with open(classification_report_path_txt, "w") as f:
            f.write(class_report)        
        save_txt_as_image(classification_report_path_txt, classification_report_path)
        save_confusion_matrix(conf_matrix, labels=["0", "1"], output_path=confusion_matrix_path)
        save_roc_curve(y_test, y_pred_proba, output_path=roc_curve_path)
        return {
            "classification_report": f"static/images/generated_classification_report.jpg",
            "confusion_matrix": f"static/images/generated_confusion_matrix.jpg",
            "roc_auc": f"static/images/generated_roc_auc.jpg",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
