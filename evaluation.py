from tsai.all import *
import numpy as np


# Define the model architecture
model = LSTM(c_out=2, n_layers=3, hidden_size=32, bidirectional= False, fc_dropout=0.2, rnn_dropout=0.2)
net_dict = torch.load('models/28-07-2024-19-44-31.pth')

# Load the model's state dictionary
model.load_state_dict(net_dict)

# Predict on the validation set
valid_dl = dls.valid
preds, targets = learn.get_preds(dl=valid_dl)
print(f"Predictions: {preds}")
print(f"Targets: {targets}")

# Convert predictions to class labels if needed
pred_labels = np.argmax(preds, axis=1)
print(f"Predicted class labels: {pred_labels}")

# Example: Predict on a new batch of data
new_data = X[splits[1]][:5]  # Using the first 5 samples from the validation set
new_preds = learn.predict(new_data)
print(f"New predictions: {new_preds}")