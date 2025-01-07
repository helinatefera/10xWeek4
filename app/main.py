from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import numpy as np

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50)  # Initialize hidden state
        c_0 = torch.zeros(2, x.size(0), 50)  # Initialize cell state

        out, _ = self.lstm(x, (h_0, c_0))  # LSTM layer
        out = self.fc(out[:, -1, :])  # Fully connected layer
        return out

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "../sales_prediction_model.pth"
model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# HTML template embedded as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
        h1 { color: #333; }
        input { padding: 10px; width: 300px; margin-right: 10px; }
        button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        #result { margin-top: 20px; font-size: 18px; }
    </style>
    <script>
        async function getPrediction() {
            const features = document.getElementById("features").value;
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: `features=${features}`
            });
            const result = await response.json();
            const resultDiv = document.getElementById("result");
            if (result.error) {
                resultDiv.innerText = `Error: ${result.error}`;
                resultDiv.style.color = "red";
            } else {
                resultDiv.innerText = `Predicted Sales: ${result.prediction}`;
                resultDiv.style.color = "green";
            }
        }
    </script>
</head>
<body>
    <h1>Sales Prediction</h1>
    <div>
        <label for="features">Enter Sales (comma-separated):</label>
        <input type="text" id="features" placeholder="5000,5200,5300">
        <button onclick="getPrediction()">Predict</button>
    </div>
    <div id="result"></div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        input_data = request.form["features"]
        input_list = list(map(float, input_data.split(",")))

        # Preprocess the input
        features = np.array(input_list, dtype=np.float32)
        features = torch.tensor(features).view(1, -1, 1)  # Reshape for LSTM

        # Make prediction
        with torch.no_grad():
            prediction = model(features)

        return jsonify({"prediction": round(prediction.item(), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
