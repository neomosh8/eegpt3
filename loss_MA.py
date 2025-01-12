import pandas as pd
import matplotlib.pyplot as plt

# Define a function to calculate the moving average
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Read the data file
file_path = 'log.txt'  # Replace with your file path
data = []

# Parse the file
with open(file_path, 'r') as file:
    for line in file:
        parts = line.split()
        epoch = int(parts[0])
        data_type = parts[1]
        value = float(parts[2])
        if data_type == 'train':
            data.append([epoch, value])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Epoch', 'Train Loss'])

# Calculate the moving average (using a window size of 50 for demonstration)
window_size = 200
df['Moving Average'] = moving_average(df['Train Loss'], window_size)

# Plot the training loss and its moving average
plt.figure(figsize=(12, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', alpha=0.6)
plt.plot(df['Epoch'], df['Moving Average'], label=f'{window_size}-Epoch Moving Average', linewidth=2)
plt.title('Training Loss and Moving Average')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
