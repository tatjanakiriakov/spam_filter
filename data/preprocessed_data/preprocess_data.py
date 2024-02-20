import pandas as pd

df = pd.read_csv("/Users/tatjanakiriakov/Desktop/spam_filter/data/raw_data/SMSSpamCollection.csv", on_bad_lines="skip", sep=";", names=["label", "message"])

df.head()

df = df.drop(df.index[0])

df.head()

df = df.dropna(subset=['message'])
df['message_length'] = df['message'].apply(lambda x: len(str(x)))
average_length = df['message_length'].mean()
print("Average character length of the 'message' column:", average_length)

df['message_length'] = df['message'].str.len()

bins = [0, 50, 100, 150, 200, 250, 300]


distribution = pd.cut(df['message_length'], bins=bins).value_counts()

print("Distribution of average character length in column 'message':")
print(distribution)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)

X = vectorizer.fit_transform(df['message'])

feature_names = vectorizer.get_feature_names_out()

word_counts = X.sum(axis=0)

word_freq_df = pd.DataFrame(word_counts, columns=feature_names).transpose()
word_freq_df.columns = ['Frequency']
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

print("Top 10 most frequent words:")
print(word_freq_df.head(10))


# In[58]:


label_counts = df['label'].value_counts()

# Print the counts of "spam" and "ham" values
print("Number of 'spam' values:", label_counts['spam'])
print("Number of 'ham' values:", label_counts['ham'])


# In[59]:


spam_messages = df[df['label'] == 'spam']['message']
ham_messages = df[df['label'] == 'ham']['message']

vectorizer_spam = CountVectorizer(stop_words='english', max_features=1000)
X_spam = vectorizer_spam.fit_transform(spam_messages)

feature_names_spam = vectorizer_spam.get_feature_names_out()

word_counts_spam = X_spam.sum(axis=0)

word_freq_df_spam = pd.DataFrame(word_counts_spam, columns=feature_names_spam).transpose()
word_freq_df_spam.columns = ['Frequency']
word_freq_df_spam = word_freq_df_spam.sort_values(by='Frequency', ascending=False)

print("Top 10 most frequent words in 'spam' messages:")
print(word_freq_df_spam.head(10))

vectorizer_ham = CountVectorizer(stop_words='english', max_features=1000)
X_ham = vectorizer_ham.fit_transform(ham_messages)

feature_names_ham = vectorizer_ham.get_feature_names_out()

word_counts_ham = X_ham.sum(axis=0)

word_freq_df_ham = pd.DataFrame(word_counts_ham, columns=feature_names_ham).transpose()
word_freq_df_ham.columns = ['Frequency']
word_freq_df_ham = word_freq_df_ham.sort_values(by='Frequency', ascending=False)

print("\nTop 10 most frequent words in 'ham' messages:")
print(word_freq_df_ham.head(10))


# In[63]:


duplicates = df[df.duplicated()]

print("Duplicate Rows:")
print(duplicates)


df_no_duplicates = df.drop_duplicates()

print("Shape of DataFrame before removing duplicates:", df.shape)
print("Shape of DataFrame after removing duplicates:", df_no_duplicates.shape)

output_file = "/Users/tatjanakiriakov/Desktop/spam_filter/data/processed_data/preprocess_data.csv"

df_no_duplicates.to_csv(output_file, index=False)

print("Changes saved to", output_file)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = df.dropna(subset=['message'])

all_messages = ' '.join(df['message'])

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform([all_messages])

feature_names = vectorizer.get_feature_names_out()

word_freq = dict(zip(feature_names, X.toarray()[0]))

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Most Frequent Words')
plt.axis('off')
plt.show()


import matplotlib.pyplot as plt

df['message_length'] = df['message'].apply(len)

plt.figure(figsize=(10, 6))
plt.hist(df['message_length'], bins=50, color='lightgreen', edgecolor='black')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.title('Distribution of Message Length')
plt.grid(True)
plt.show()
