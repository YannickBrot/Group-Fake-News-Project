import pandas as pd
import matplotlib.pyplot as plt

#path = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'  # Placeholder URL
path = '995,000_rows.csv'  # Placeholder path

# Read the CSV file into a DataFrame
df = pd.read_csv(path)

# Initialize dictionaries to hold total word lengths and total word counts for each article type
total_word_lengths = {}
total_word_counts = {}

print('starting loop')
# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Get the article type
    arttype = row['type']

    # Ensure the content is a string to avoid errors with missing or NaN values
    content = str(row['content'])

    # Split the content into words, filtering out empty strings
    words = [word for word in content.split() if word]

    # Calculate the total length of all words in the content
    total_length = sum(len(word) for word in words)
    word_count = len(words)

    # Update the total word lengths and counts for the article type
    if arttype in total_word_lengths:
        total_word_lengths[arttype] += total_length
        total_word_counts[arttype] += word_count
    else:
        total_word_lengths[arttype] = total_length
        total_word_counts[arttype] = word_count

print('finished loop')

# Calculate the average word length for each article type
average_word_lengths = {arttype: total_word_lengths[arttype] / total_word_counts[arttype]
                        for arttype in total_word_lengths}
sorted_average_word_lengths = sorted(average_word_lengths.items(), key=lambda item: item[1], reverse=True)

types = [str(k) for k, v in sorted_average_word_lengths]
averages = [v for k, v in sorted_average_word_lengths]

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(types, averages)
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Prevent labels from overlapping
plt.show()
