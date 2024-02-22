import pandas as pd
import re

# URL of the CSV file
url = '995,000_rows.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(url)

# Initialize an empty dictionary to hold the frequencies
truth_frequencies = {}

# Compile a regular expression pattern for improved performance
# The pattern looks for 'true', 'truth', and also covers variations like 'truly', 'truthful', etc.
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

    # If the article type is already in the dictionary, add the count
    if arttype in truth_frequencies:
        truth_frequencies[arttype] += true_count
    else:
        # If the article type is not in the dictionary, initialize it with the count
        truth_frequencies[arttype] = true_count

print('finished loop')
# Display the truth frequencies dictionary
print(truth_frequencies)
