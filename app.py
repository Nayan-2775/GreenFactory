from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn

# Define the model class
class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.layer1 = nn.Linear(9, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Load the model
model = EnergyModel()
model.load_state_dict(torch.load('energy_model.pth', weights_only=True))
model.eval()

# Create Flask app
app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = torch.FloatTensor([[
        data['year'], data['month'], data['solar'], data['wind'],
        data['bioenergy'], data['hydro'], data['geothermal'],
        data['battery_storage'], data['energy_demand']
    ]])
    prediction = model(input_data).item()
    return jsonify({'predicted_total_energy': prediction})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
