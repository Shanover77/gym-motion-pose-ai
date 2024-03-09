import os
import pandas as pd

# Set the directory path
directory_path = 'trainable_data'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Check if there are any CSV files in the directory
if not csv_files:
    print("No CSV files found in the directory.")
else:
    # Initialize an empty DataFrame to store the concatenated data
    concatenated_data = pd.DataFrame()

    # Concatenate each CSV file into the DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        data = pd.read_csv(file_path)
        concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)

    clen = len(csv_files)
    # Save the concatenated data to a new CSV file
    output_file_path = f'exer_{clen}_pipe_apexdata_w16_deg.csv'
    concatenated_data.to_csv(output_file_path, index=False)

    print(f"Concatenation complete. Result saved to {output_file_path}")
