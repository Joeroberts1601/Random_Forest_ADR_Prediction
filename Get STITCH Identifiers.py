import pandas as pd
import os

# File paths
file1 = "All_filtered_chemicals_file.tsv"
file2 = "Drug InChi Keys/SSRIs Inhibitors Inchi.csv"
output_folder = "STITCH_Identifiers"
output_file = os.path.join(output_folder, "matched_chemicals.csv")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the data
df1 = pd.read_csv(file1, sep="\t")  # Assuming TSV format
df2 = pd.read_csv(file2)

# Normalize 'name' and 'Drug' columns (lowercase, trimmed)
df1['name_normalized'] = df1['name'].str.lower().str.strip()
df2['Drug_normalized'] = df2['Drug'].str.lower().str.strip()

# Perform the merge based on normalized values
merged_df = pd.merge(df1, df2, left_on='name_normalized', right_on='Drug_normalized', how='inner')

# Extract the 'chemical' and 'Drug' columns from the matches
result_df = merged_df[['chemical', 'Drug']]

# Save the results to a CSV file
result_df.to_csv(output_file, index=False)

print(f"Matched chemicals with drug names saved to: {output_file}")
