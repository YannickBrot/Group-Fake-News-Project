# Group-Fake-News-Project

by Max Meldal & Yannick Brot Christensen

## Prerequisites:

### Place the following file in the root folder of the project:
- '995,000_rows.csv', csv file of articles from FakeNewsCorpus
- 'glove.6B.100d.txt' - zip file can be downloaded from [Stanford Glove](https://nlp.stanford.edu/projects/glove/), by clicking glove.6B.zip. txt file is in zip.
- 'scraped_articles_comma.csv' - csv file containing articles scraped from BBC news. File should have only column _content_ and each row should have the contents of the article.

### pip install
While standing in the root folder run following command: \
$ pip install -r requirements.txt

## Reproduce results:
- Step 1: run all the cells in 'Part1.ipynb', in order from top to bottom
- Step 2: run all the cells in 'Part234.ipynb', in order from top to bottom
- Graphs and diagrams are automatically produced.
### BEWARE
Running these to files may take a long time. 'Part1.ipynb' may take from 1 to multiple hours. 'Part2.ipynb' is the same.