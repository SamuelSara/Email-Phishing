import os
import pandas as pd
import email
import quopri
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# Ensure NLTK resources are downloaded (for text preprocessing)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase, remove punctuation, tokenize, and remove stopwords
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def extract_text_from_eml(file_path):
    # Extract text content from .eml file
    with open(file_path, 'r', encoding='utf-8') as file:
        msg = email.message_from_file(file)
        text = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    text = part.get_payload(decode=True)
                    try:
                        text = quopri.decodestring(text).decode('utf-8')
                    except:
                        pass
                    break
        else:
            text = msg.get_payload(decode=True)
            try:
                text = quopri.decodestring(text).decode('utf-8')
            except:
                pass
        return text

if 'src' not in os.getcwd():
    os.chdir('C:\\Users\\samsa\\OneDrive\\Desktop\\New folder\\New-Folder\\Email\\src')

# Load and prepare training data
df = pd.read_csv('Phishing_Email.csv')
df.dropna(inplace=True)
df['Email Text'] = df['Email Text'].apply(preprocess_text)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Email Type'])
X_train, X_test, y_train, y_test = train_test_split(df['Email Text'], y, test_size=0.2)

# Build and train the model
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipeline.fit(X_train, y_train)

# Save the trained model and label encoder to disk
joblib.dump(pipeline, 'email_classifier_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Load the model and label encoder
model = joblib.load('email_classifier_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def predict_eml(file_path):
    # Predict whether an .eml file is phishing or not
    eml_text = extract_text_from_eml(file_path)
    preprocessed_text = preprocess_text(eml_text)
    prediction = model.predict([preprocessed_text])
    return label_encoder.inverse_transform(prediction)[0]

# Example usage
eml_file_path = 'Safe.eml'  # Assuming 'Safe.eml' is in the current working directory
result = predict_eml(eml_file_path)
print(f"The email is classified as {result}")
