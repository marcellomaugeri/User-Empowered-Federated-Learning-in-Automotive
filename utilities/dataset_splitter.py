import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_csv, output_csv1, output_csv2, output_csv3):
    df = pd.read_csv(input_csv)
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]
    
    # 80/20 split
    features_80, features_20, labels_80, labels_20 = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=1337)
    
    # 40/40/20 split (by splitting the above 80%)
    features_40_1, features_40_2, labels_40_1, labels_40_2 = train_test_split(
        features_80, labels_80, test_size=0.50, stratify=labels_80, random_state=322)
    
    df_40_1 = pd.concat([labels_40_1.reset_index(drop=True), features_40_1.reset_index(drop=True)], axis=1)
    df_40_2 = pd.concat([labels_40_2.reset_index(drop=True), features_40_2.reset_index(drop=True)], axis=1)
    df_20 = pd.concat([labels_20.reset_index(drop=True), features_20.reset_index(drop=True)], axis=1)
    df_40_1.to_csv(output_csv1, index=False)
    df_40_2.to_csv(output_csv2, index=False)
    df_20.to_csv(output_csv3, index=False)

    print("Done!")

# Usage
input_csv = 'Scaled_EngineFaultDB_Final.csv'
output_csv1 = 'train_1.csv'
output_csv2 = 'train_2.csv'
output_csv3 = 'test_1.csv'

split_dataset(input_csv, output_csv1, output_csv2, output_csv3)