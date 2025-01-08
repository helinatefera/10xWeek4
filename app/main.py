import torch
import torch.nn as nn
from flask import Flask, render_template_string, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and its weights
model = LSTMModel(input_size=4, hidden_size=50, num_layers=2)  # Define the architecture
model.load_state_dict(torch.load('../sales_prediction_model.pth'))  # Load the saved model weights
model.eval()  # Set the model to evaluation mode

# Define the MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

# Sample data for the scaler (you can replace this with your actual dataset)
sample_data = np.array([[124, 0, 0, 0], [87, 0, 0, 0], [74, 778, 0, 0], [0, 0, 0, 0], [76, 1002, 0, 0]])

# Fit the scaler on the sample data (replace with your actual dataset)
scaler.fit(sample_data)

# Home route
@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Prediction</title>
    </head>
    <body>
        <h1>Sales Prediction</h1>
        <form action="/predict" method="post">
            <label for="customers">Customers:</label>
            <input type="text" id="customers" name="customers" required><br><br>
            
            <label for="days_to_holiday">Days to Holiday:</label>
            <input type="text" id="days_to_holiday" name="days_to_holiday" required><br><br>
            
            <label for="days_after_holiday">Days after Holiday:</label>
            <input type="text" id="days_after_holiday" name="days_after_holiday" required><br><br>
            
            <label for="is_weekend">Is Weekend (0 or 1):</label>
            <input type="text" id="is_weekend" name="is_weekend" required><br><br>
            
            <input type="submit" value="Predict">
        </form>
        
        {% if prediction %}
            <h3>Predicted Sales: {{ prediction }}</h3>
        {% endif %}
    </body>
    </html>
    """)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    customers = float(request.form['customers'])
    days_to_holiday = float(request.form['days_to_holiday'])
    days_after_holiday = float(request.form['days_after_holiday'])
    is_weekend = int(request.form['is_weekend'])

    # Normalize the input data
    input_data = np.array([[customers, is_weekend, days_to_holiday, days_after_holiday]])
    input_data = scaler.transform(input_data)

    # Reshape the input for LSTM [batch_size, seq_length, input_size]
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_data)

    # Get the predicted sales value
    predicted_sales = prediction.item()

    # Return the result
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Prediction</title>
    </head>
    <body>
        <h1>Sales Prediction</h1>
        <form action="/predict" method="post">
            <label for="customers">Customers:</label>
            <input type="text" id="customers" name="customers" required><br><br>
            
            <label for="days_to_holiday">Days to Holiday:</label>
            <input type="text" id="days_to_holiday" name="days_to_holiday" required><br><br>
            
            <label for="days_after_holiday">Days after Holiday:</label>
            <input type="text" id="days_after_holiday" name="days_after_holiday" required><br><br>
            
            <label for="is_weekend">Is Weekend (0 or 1):</label>
            <input type="text" id="is_weekend" name="is_weekend" required><br><br>
            
            <input type="submit" value="Predict">
        </form>
        
        <h3>Predicted Sales: {{ predicted_sales }}</h3>
    </body>
    </html>
    """, predicted_sales=predicted_sales)

if __name__ == '__main__':
    app.run(debug=True)
