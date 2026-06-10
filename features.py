
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("STAGE 2: (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
X_features = tfidf.fit_transform(df['cleaned_text'])
print(f" Feature matrix shape: {X_features.shape}")

# STAGE 3: 
print("STAGE 3: Generating Risk/Sentiment labels using VADER...")
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def assign_risk_label(text):
    score = sia.polarity_scores(text)['compound']
    if score <= -0.05:
        return 0  # Class 0: High Risk / Negative
    elif score >= 0.05:
        return 2  # Class 2: Low Risk / Positive
    else:
        return 1  # Class 1: Medium Risk / Neutral

df['Target'] = df['cleaned_text'].apply(assign_risk_label)
y_labels = df['Target']
print("\nLabel Distribution:")
print(df['Target'].value_counts().rename(index={0: 'High Risk (0)', 1: 'Medium Risk (1)', 2: 'Low Risk (2)'}))

