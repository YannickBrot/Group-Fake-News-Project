import pandas as pd
import re
import matplotlib.pyplot as plt

# Assuming url is correctly defined and points to a valid CSV file
#path = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'  # Placeholder URL
path = '995,000_rows.csv'  # Placeholder URL

# Read the CSV file into a DataFrame
df = pd.read_csv(path)

# Initialize an empty dictionary to hold the frequencies and article counts
truth_frequencies = {}
article_counts = {}

# Compile a regular expression pattern for improved performance
pattern = re.compile(r'\btr(?:ue|uth|uly|uthful|uthfulness)\b', re.IGNORECASE)

print('starting loop')
# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Get the article type
    arttype = row['type']

    # Ensure the content is a string to avoid errors with missing or NaN values
    content = str(row['content'])

    # Use re.findall() to find all occurrences of the pattern
    matches = pattern.findall(content)

    # The number of occurrences is the length of the list returned by findall()
    true_count = len(matches)

    # If the article type is already in the dictionary, update counts
    if arttype in truth_frequencies:
        truth_frequencies[arttype] += true_count
        article_counts[arttype] += 1
    else:
        # If the article type is not in the dictionary, initialize it
        truth_frequencies[arttype] = true_count
        article_counts[arttype] = 1

print('finished loop')

# Calculate the weighted truth frequencies
weighted_truth_frequencies = {arttype: truth_frequencies[arttype] / article_counts[arttype] for arttype in
                              truth_frequencies}
sorted_truth_frequencies = sorted(weighted_truth_frequencies.items(), key=lambda item: item[1], reverse=True)

types = [str(k) for k, v in sorted_truth_frequencies]
frequencies = [v for k, v in sorted_truth_frequencies]

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(types, frequencies)
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Prevent labels from overlapping
plt.show()
