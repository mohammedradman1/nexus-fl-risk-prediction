import pandas as pd

# Here, we create a smaple dataset and save it
data = {
    "location_id": ["A1", "A2", "A3", "A4", "A5"],
    "temperature": [35.5, 38.2, 36.0, 33.4, 39.1],
    "rainfall": [120.0, 85.5, 95.0, 70.2, 130.3],
    "humidity": [60, 55, 65, 50, 70],
    "wind_speed": [10.5, 12.3, 9.8, 11.1, 13.0],
    "flood_label": [1, 1, 0, 0, 1]  # 1 = flood occurred, 0 = no flood
}

df = pd.DataFrame(data)

# Save to data/sample_input.csv
output_path = "sample_input.csv"
df.to_csv(output_path, index=False)

output_path
# The output path is where the sample dataset is saved.