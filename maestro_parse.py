import pandas as pd
import shutil
import os
import csv

# Load the CSV file into a pandas DataFrame
maestro_path = "/Users/liampilarski/Downloads/maestro-v3.0.0/"

df = pd.read_csv(maestro_path + 'maestro-v3.0.0.csv')
print(df.values)

# Filter the rows where the artist name is "Johann Sebastian Bach"
bach_rows = df[df['canonical_composer'] == 'Johann Sebastian Bach']

# Ensure the output directory exists
if not os.path.exists('output'):
    os.makedirs('output')

# Iterate over the filtered rows and copy files
for index, row in bach_rows.iterrows():
    file_path = maestro_path + row['midi_filename']
    if os.path.exists(file_path):
        shutil.copy(file_path, 'output')
    else:
        print(f"File {file_path} not found!")
