import pandas as pd
import re
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

# STAGE 1: DATA EXTRACTION & PREPROCESSING
print("STAGE 1: Loading and Preprocessing Data...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
print("Initializing dataset stream...")
streamed_dataset = load_dataset("winterForestStump/10-K_sec_filings", split="001", streaming=True)
target_rows = 2500
data_list = []
print(f"Fetching and cleaning {target_rows} filings on the fly...")
for i, example in enumerate(tqdm(streamed_dataset, total=target_rows)):
    if i >= target_rows:
        break
    # Combine 'text', 'Business', and 'Risk Factors' to ensure content if 'text' is empty
    raw_text = str(example.get('text', '') or \
                   example.get('Business', '') or \
                   example.get('Risk Factors', '') or \
                   example.get('Unresolved Staff Comments', '') or \
                   example.get('Properties', '') or \
                   example.get('Legal Proceedings', '') or \
                   example.get('Mine Safety Disclosures', '') or \
                   example.get('Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer P', ''))[:15000]
    text = re.sub(r'<[^>]+>', ' ', raw_text) 
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower() 
    words = text.split()
    cleaned_text = " ".join([w for w in words if w not in stop_words and len(w) > 2])
    data_list.append({
        'cleaned_text': cleaned_text
    })
df = pd.DataFrame(data_list)
print(f" Successfully loaded and cleaned {len(df)} rows!\n")
