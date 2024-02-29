import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate the LIX number
def calculate_lix(text):
    words = text.split()
    num_words = len(words)
    num_sentences = text.count('.') + text.count('?') + text.count('!')
    num_long_words = sum(1 for word in words if len(word) > 6)
    if num_sentences == 0:  # Avoid division by zero
        return 0
    lix = (num_words / num_sentences) + (num_long_words / num_words) * 100
    return lix

# Load your dataset
path = '995,000_rows.csv'
df = pd.read_csv(path)

# Calculate the LIX number for each content
df['lix_number'] = df['content'].apply(lambda x: calculate_lix(str(x)))

# Group by 'type' and calculate the average LIX number for each type
avg_lix_number_by_type = df.groupby('type')['lix_number'].mean()

# Plotting
plt.figure(figsize=(15, 6))
avg_lix_number_by_type.plot(kind='bar')
plt.title('Average LIX Number by Type')
plt.xlabel('Type')
plt.ylabel('Average LIX Number')
plt.xticks(rotation=45)
plt.show()
