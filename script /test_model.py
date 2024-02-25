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
            
df = pd.read_csv("/Users/tatjanakiriakov/Desktop/spam_filter/data/raw_data/SMSSpamCollection.csv",  
                 sep=";", 
                 names=["label", "message"])

df = df.dropna(subset=['message'])

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

model, vectorizer = train_model(X_train, y_train)

test_predicted_labels = classify_messages(model, vectorizer, X_test)

for message, label in zip(X_test, test_predicted_labels):
    if label == 'ham':
        print("Message passed the low-risk level:", message)
    else:
        move_high_risk_messages([message], "/Users/tatjanakiriakov/Desktop/spam_filter/models/high_risk/high_risk.txt")


# Define some random messages
random_messages = [
    "I think the product has many great advantages. You should definitely consider buying it!",
    "I would like to give my feedback to the product XY. I think the customer service was great and the delivery was fast and unproblematic.",
    "Hey! Hope you're having a great day. Just wanted to check in and see how you're doing. Let's catch up soon!",
    "You've won a free luxury vacation to your dream destination! Claim your prize now by providing your personal details."]
    
# Classify the random messages using the trained model and vectorizer
predicted_labels = classify_messages(model, vectorizer, random_messages)

# Print the predicted labels for each random message
for message, label in zip(random_messages, predicted_labels):
    if label == 'ham':
        print("Message passed the low-risk level:", message)
    else:
        print("Message classified as high-risk:", message)




