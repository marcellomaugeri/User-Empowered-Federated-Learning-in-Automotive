import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_dataset(input_csv, output_csv):
    # Read the dataset from the CSV file
    df = pd.read_csv(input_csv)

    # Strip the first column (assuming it's the label)
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:]

    # Data scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Conversion to a DataFrame again
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    scaled_all = pd.concat([labels.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

    # Save the scaled dataset to a new CSV file
    scaled_all.to_csv(output_csv, header=None, index=False)
    print(f"Scaled dataset saved to {output_csv}")

# Conversion
input_csv = 'EngineFaultDB_Final.csv'
output_csv = 'Scaled_EngineFaultDB_Final.csv'
scale_dataset(input_csv, output_csv)