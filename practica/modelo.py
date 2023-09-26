import nltk
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.linear_model import LogisticRegression


nltk.download('punkt')


def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



keywords = ['terremoto', 'huracán', 'inundación', 'flood', 'hurricane']


# Función para etiquetar los tweets
def label_tweet(tweet):
    for word in keywords:
        if word in tweet.lower():
            return 1  
    return 0  



df = pd.read_csv('datos_combinados.tsv',sep='\t')    


df['is_disaster'] = df['tweet_text'].apply(label_tweet)

label_counts = df['is_disaster'].value_counts()


print(label_counts)



corpus = [nltk.word_tokenize(preprocess(text)) for text in df['tweet_text']]

#  Word2Vec
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
model.train(corpus, total_examples=len(corpus), epochs=10)


def text_to_vector(text, model):
    words = nltk.word_tokenize(preprocess(text))
    vectors = [model.wv[word] for word in words if word in model.wv.index_to_key]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

#  vectores para cada tweet
vectors = [text_to_vector(text, model) for text in df['tweet_text']]

# Etiquetas
labels = df['is_disaster'].tolist()

#  entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42)

# Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#  predicciones 
y_pred = clf.predict(X_test)


print(classification_report(y_test, y_pred))


#  Support Vector Machine
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print("Resultados usando SVM:")
print(classification_report(y_test, y_pred_svm))

# Logistic Regression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
print("Resultados usando Regresión Logística:")
print(classification_report(y_test, y_pred_lr))

