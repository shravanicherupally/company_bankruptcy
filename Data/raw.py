import pandas as pd

# Step 1: Load raw dataset
df = pd.read_csv("dataset.csv")

# Step 2: Clean data
# Drop unnecessary columns (like 'year' if present)
if 'year' in df.columns:
    df = df.drop(columns=['year'])

print("✅ Columns in dataset:", df.columns.tolist())

# Step 3: Handle missing values
if df.isnull().sum().any():
    print("⚠️ Missing values detected. Filling with column mean.")
    df.fillna(df.mean(), inplace=True)
else:
    print("✅ No missing values.")

# Step 4: Encode labels if needed
if df['status_label'].dtype == 'object':
    df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})

# Debug: check class counts
print("🔍 Unique values in 'status_label':", df['status_label'].unique())
print("🔢 Class distribution:")
print(df['status_label'].value_counts())

# Step 5: Balance dataset
df_alive = df[df['status_label'] == 0]
df_failed = df[df['status_label'] == 1]

if len(df_alive) == 0 or len(df_failed) == 0:
    raise ValueError("❌ No 'alive' or 'failed' data found after filtering. Please check label encoding.")

df_alive_sample = df_alive.sample(n=len(df_failed), random_state=42)
df_balanced = pd.concat([df_alive_sample, df_failed], axis=0)

# Step 6: Shuffle after balancing
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 7: Select top 4 features and target (no 'row_id')
selected_columns = ['X1', 'X4', 'X6', 'X10', 'status_label']
missing = set(selected_columns) - set(df_balanced.columns)
if missing:
    raise ValueError(f"❌ Missing columns in your CSV: {missing}")

df_selected = df_balanced[selected_columns]

# Step 8: Save final dataset
output_path = "company_top4_features.csv"
df_selected.to_csv(output_path, index=False)
print(f"✅ Cleaned and balanced dataset saved to: {output_path}")
