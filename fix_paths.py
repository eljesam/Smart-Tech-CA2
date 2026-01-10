import pandas as pd

# Load the formatted CSV
df = pd.read_csv('data/driving_log_formatted.csv')

def clean_path(path):
    if isinstance(path, str):
        # Convert Windows backslashes and extract filename
        filename = path.replace("\\", "/").split("/")[-1].strip()
        return f"IMG/{filename}"
    return path

# Apply to image columns
for col in ['center', 'left', 'right']:
    df[col] = df[col].apply(clean_path)

# Save cleaned CSV
df.to_csv('data/driving_log_relative.csv', index=False)

print(" Image paths converted to relative paths")
print(df[['center', 'left', 'right']].head())