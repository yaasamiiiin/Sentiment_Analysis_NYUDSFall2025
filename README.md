# Sentiment_Analysis_NYUDSFall2025

## Project links:
* [Kaggle Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset/code)
* [Mid-Term Presentation (Google Slides)](https://docs.google.com/presentation/d/1AmmNKCUKoWsUOTmKg2sdAeAwAfl-iAcbBd89os0nras/edit?slide=id.p#slide=id.p)
* [Final Presentation (Google Slides)](https://docs.google.com/presentation/d/1VButAovGpUdE85XRTmdxm-TjSFR8bFRSqlTEQ8BZfWE/edit?slide=id.p#slide=id.p)

#### How to install the source codes:

After pulling the github repo, do the following:

```bash
# First run this in the terminal
pip install -e .
```
This installs the setup.py file in the directory.
Then add this code block in the beginning of your script:
```python
# Setup cell - Run this first
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import from src
from src import load_data, clean_data, map_sentiments

print("✓ Setup complete!")
```

#### Exmaple usage:
Loading and cleaning dataset:
```python
# Load and clean data
df = load_data('../Data/sentimentdataset.csv')
df_clean = clean_data(df)
```
