import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer

def classify_messages(model, vectorizer, messages):
    message_vectors = vectorizer.transform(messages)
    predicted_labels = model.predict(message_vectors)
    return predicted_labels

def move_high_risk_messages(messages, file_path):
    with open(file_path, "a") as file:
        for message in messages:
            file.write(message + "\n")

# Read the dataset
df = pd.read_csv("/Users/tatjanakiriakov/Desktop/spam_filter/data/raw_data/SMSSpamCollection.csv",  
                 sep=";", 
                 names=["label", "message"])

df = df.dropna(subset=['message'])

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Train the model
model, vectorizer = train_model(X_train, y_train)

# Classify messages in the testing data
test_predicted_labels = classify_messages(model, vectorizer, X_test)

for message, label in zip(X_test, test_predicted_labels):
    if label == 'ham':
        print("Message passed the low-risk level:", message)
    else:
        move_high_risk_messages([message], "/Users/tatjanakiriakov/Desktop/spam_filter/models/high_risk/high_risk.txt")

