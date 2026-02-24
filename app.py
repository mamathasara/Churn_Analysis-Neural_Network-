import gradio as gr
import pandas as pd
import pickle
import tensorflow as tf

# Load model and preprocessing components
MODEL_PATH = "models/best_churn_keras_model.keras"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/selected_features.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))
selected_features = pickle.load(open(FEATURES_PATH, "rb"))

def predict_churn(online_backup, paperless_billing, payment_method, monthly_charges, total_charges):
    
    # Extract numeric value from "0 - Yes"
    online_backup = int(str(online_backup).split("-")[0].strip())
    payment_method = int(str(payment_method).split("-")[0].strip())
    paperless_billing = 1 if paperless_billing else 0

    input_data = pd.DataFrame(
        [[online_backup, paperless_billing, payment_method, monthly_charges, total_charges]],
        columns=selected_features
    )

    input_scaled = scaler.transform(input_data)
    probability = float(model.predict(input_scaled, verbose=0)[0][0])

    churn_status = "Yes" if probability >= 0.5 else "No"
    return churn_status, round(probability, 4)

interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Radio(
         ["0 - Yes","1 - No","2 - No Internet Service"],
        label="Online Backup",
        value="0 - Yes"
        ),
        gr.Checkbox(label="Paperless Billing (Checked = Yes, Unchecked = No)"),
        gr.Radio([
            "0 - Electronic Check",
            "1 - Mailed Check",
            "2 - Bank Transfer (Automatic)",
            "3 - Credit Card (Automatic)"
        ], label="Payment Method",value="0 - Electronic Check"),
        gr.Slider(0,150,step=0.1,label="Monthly Charges", value=50),
        gr.Slider(0,10000,step=0.1,label="Total Charges", value=1000)
    ],
    outputs=[
        gr.Textbox(label="Churn Status"),
        gr.Number(label="Churn Probability")
    ],
    title="Customer Churn Prediction (Neural Network)",
    description="Predict whether a telecom customer will churn using a trained Neural Network model.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    interface.launch()