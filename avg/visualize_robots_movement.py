import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')

x_coords = data['px']
y_coords = data['py']

unique_stations = data['goal_status'].unique()

# Extract 'stopped at' labels and retrieve station names
stopped_labels = [label for label in unique_stations if "stopped at" in label]
station_names = [label.split("stopped at ")[1] for label in stopped_labels]
station_coords = {name: data[data['goal_status'] == label].iloc[0][['px', 'py']] for name, label in zip(station_names, stopped_labels)}

unknown_stopped_coords = data[data['goal_status'] == 'stopped (unknown)'][['px', 'py']]

plt.figure(figsize=(14, 10))
plt.plot(x_coords, y_coords, '-o', markersize=2, lw=1, alpha=0.5, color='grey')
plt.scatter(x_coords.iloc[0], y_coords.iloc[0], c='green', s=100, label='Start')
plt.scatter(x_coords.iloc[-1], y_coords.iloc[-1], c='red', s=100, label='End')

# Plot points for "stopped (unknown)" status
plt.scatter(unknown_stopped_coords['px'], unknown_stopped_coords['py'], c='purple', s=100, label='Stopped (Unknown)')

# Add station labels with increased font size and adjusted positions
label_font_size = 18
for station, coords in station_coords.items():
    plt.scatter(coords['px'], coords['py'], s=100)
    
    offset_x, offset_y = 0.07, 0.07
    plt.text(coords['px'] + offset_x, coords['py'] + offset_y, station, fontsize=label_font_size, ha='right')

plt.title('Robot Movement across the Factory Floor with Station Labels')
plt.xlabel('X Coordinate (px)')
plt.ylabel('Y Coordinate (py)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

