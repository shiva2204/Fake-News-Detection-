import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Datasets
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# 2. Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# 3. Combine and shuffle
df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

# 4. Define Features (X) and Target (y)
x = df['text']
y = df['label']

# 5. Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# 6. Initialize TfidfVectorizer (Turns text into math)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# 7. Initialize and Train PassiveAggressiveClassifier
# This is great for massive text datasets
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 8. Evaluation
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'✅ Accuracy: {round(score*100,2)}%')

# 9. Quick Test Function
def predict_news(text):
    data = tfidf_vectorizer.transform([text])
    prediction = pac.predict(data)
    print(f'Prediction: {prediction[0]}')

# Try it out
sample_news = "The moon is made of green cheese and NASA has been hiding it for years."
predict_news(sample_news)
