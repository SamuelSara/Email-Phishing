import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
file_path = 'C:/Users/samsa/OneDrive/Desktop/New folder/New-Folder/Email/src/Phishing_Email.csv'
df = pd.read_csv(file_path)

# Inspect the data for null values and class distribution
print(df.isnull().sum())
print(df['Email Type'].value_counts())

# Preprocess and clean your text data here (implement your cleaning function)
def clean_text(text):
    # Your cleaning process here
    return text

df['Email Text'] = df['Email Text'].apply(clean_text)

# Encode categorical labels to numerical labels
label_encoder = LabelEncoder()
df['Email Type'] = label_encoder.fit_transform(df['Email Type'])

# Split the data into features and target labels
X = df['Email Text']
y = df['Email Type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
