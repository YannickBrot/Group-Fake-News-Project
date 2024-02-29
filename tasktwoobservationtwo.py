## count average length of articles for different lables
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
path = '995,000_rows.csv'
df = pd.read_csv(path)

# Calculate the length of each content
df['content_length'] = df['content'].apply(lambda x: len(str(x)))

# Group by 'type' and calculate the average content length for each type
avg_content_length_by_type = df.groupby('type')['content_length'].mean()

# Plotting
plt.figure(figsize=(15, 6))
avg_content_length_by_type.plot(kind='bar')
plt.title('Average Content Length by Type')
plt.xlabel('Type')
plt.ylabel('Average Content Length')
plt.xticks(rotation=45)
plt.show()
