import os
import pandas as pd

# Create output folder
output_folder = "backend/reference_database/news"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_json("backend/News_Category_Dataset_v3.json", lines=True)

# Combine useful fields
df["text"] = df["headline"].fillna('') + ". " + df["short_description"].fillna('')

# Limit for performance (adjust later)
df = df.head(3000)

# Save each article as separate text file
for i, row in df.iterrows():
    with open(f"{output_folder}/article_{i}.txt", "w", encoding="utf-8") as f:
        f.write(row["text"])

print("News corpus created successfully.")