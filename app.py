from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time

app = Flask(__name__)

# Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

stop_words_id = set(stopwords.words('indonesian'))

def preprocess_document(document):
    document = document.lower()
    words = nltk.word_tokenize(document)
    words = [word for word in words if word.lower() not in stop_words_id]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Membaca dokumen
def read_documents_from_folder(folder_path):
    file_names = os.listdir(folder_path)
    corpus = []
    document_names = []  
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            document_content = file.read()
            corpus.append(document_content)
            document_names.append(file_name)
    return corpus, document_names

# Indexing & Term Weighting
folder_path = "Antara News Corpus"
corpus, document_names = read_documents_from_folder(folder_path)
preprocessed_documents = [preprocess_document(doc) for doc in corpus]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Searching
def search(query, vectorizer, tfidf_matrix, document_names):
    start_time = time.time()  
    preprocessed_query = preprocess_document(query)
    query_vector = vectorizer.transform([preprocessed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    elapsed_times = [time.time() - start_time] * len(corpus)
    non_zero_similarity_results = [
        (document_names[index], document, document.lower().count(preprocessed_query.lower()), similarity, elapsed_time)
        for index, (document, similarity, elapsed_time) in enumerate(zip(corpus, similarity_scores, elapsed_times))
        if similarity > 0
    ]

    if not non_zero_similarity_results:
        non_zero_similarity_results.append(("Not Found", "", 0, 0, 0))

    sorted_results = sorted(non_zero_similarity_results, key=lambda x: x[3], reverse=True)

    relevant_document_count = sum(1 for result in non_zero_similarity_results if result[3] > 0)

    return sorted_results, relevant_document_count

# UI/UX
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        results, relevant_document_count = search(query, vectorizer, tfidf_matrix, document_names)
        return render_template('index.html', query=query, results=results, relevant_document_count=relevant_document_count)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
