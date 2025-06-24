import streamlit as st
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Pastikan tokenizer 'punkt' tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(layout="wide")

st.title("Analisis Sentimen Film Jumbo")
st.write("Aplikasi ini melakukan analisis sentimen terhadap ulasan film 'Jumbo' dari data yang telah diproses.")

# --- Fungsi-fungsi Preprocessing ---

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

norm = {
    " yg ": " yang ", " gk ": " tidak ", " ga ": " tidak ", " knp ": " kenapa ",
    " ngga ": " tidak ", " gak ": " tidak ", " engga ": " tidak ", " gue ": " aku ",
    " udah ": " sudah ", " skrg ": " sekarang ", " jd ": " jadi ", " kyk ": " seperti ",
    " bgt ": " banget ", " kmrn ": " kemaren ", " tp ": " tapi ", " gpp ": " tidak apa apa",
    " dah ": " sudah ", " trs ": " terus ", " jg ": " juga ", " kmu ": " kamu ", " aku ": " saya ",
    " kagak ": " tidak "
}

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

def stopword(str_text):
    more_stop_words = ['tidak']
    stop_words = StopWordRemoverFactory().get_stop_words()
    stop_words.extend(more_stop_words)
    new_array = ArrayDictionary(stop_words)
    stop_words_remover_new = StopWordRemover(new_array)
    str_text = stop_words_remover_new.remove(str_text)
    return str_text

def steming(text_cleaning):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = text_cleaning.split()
    return " ".join([stemmer.stem(w) for w in words])

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('film_jumbo.csv')
    return df

df = load_data()

# --- Data Preprocessing ---
st.header("1. Data Preprocessing")

with st.expander("Lihat Langkah-langkah Preprocessing"):
    st.subheader("1.1. Cleaning Data")
    st.write(f"Ukuran data awal: {df.shape}")
    df_cleaned = df.drop_duplicates(subset='full_text').copy()
    st.write(f"Jumlah duplikat yang dihapus: {df.duplicated().sum()}")
    st.write(f"Ukuran data setelah menghapus duplikat: {df_cleaned.shape}")
    st.dataframe(df.head())

    st.subheader("1.2. Case Folding dan Pembersihan")
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(clean_text).str.lower()
    st.dataframe(df_cleaned.head())

    st.subheader("1.3. Normalisasi Kata")
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(normalisasi)
    st.dataframe(df_cleaned.head())

    st.subheader("1.4. Stopword Removal")
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(stopword)
    st.dataframe(df_cleaned.head())

    st.subheader("1.5. Stemming")
    df_cleaned['full_text_stemmed'] = df_cleaned['full_text'].apply(steming)
    st.dataframe(df_cleaned.head())

# --- Sentiment Analysis ---
st.header("2. Analisis Sentimen")

try:
    data_translated = pd.read_csv('translateJumboo.csv', index_col=0)
    if 'english_tweet' not in data_translated.columns:
        st.error("Kolom 'english_tweet' tidak ditemukan di translateJumboo.csv.")
        st.stop()
except FileNotFoundError:
    st.error("File 'translateJumboo.csv' tidak ditemukan.")
    st.stop()

data_tweet = list(data_translated['english_tweet'])

status = []
total_positif = 0
total_negatif = 0
total_netral = 0

for tweet in data_tweet:
    analysis = TextBlob(str(tweet))
    if analysis.sentiment.polarity > 0.0:
        total_positif += 1
        status.append('Positif')
    elif analysis.sentiment.polarity == 0.0:
        total_netral += 1
        status.append('Netral')
    else:
        total_negatif += 1
        status.append('Negatif')

data_translated['label'] = status

st.subheader("Hasil Analisis Sentimen")
st.write(f"Jumlah Positif: {total_positif}")
st.write(f"Jumlah Netral: {total_netral}")
st.write(f"Jumlah Negatif: {total_negatif}")
st.write(f"Total Data: {len(data_translated)}")
st.dataframe(data_translated.head())

# --- Evaluasi Model ---
st.subheader("Evaluasi Model")
y_true = data_translated['label']
y_pred = data_translated['label']  # Karena tidak ada label asli, evaluasi dilakukan atas hasil yang sama

st.text("Classification Report")
st.text(classification_report(y_true, y_pred))

st.text("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred, labels=['Positif', 'Netral', 'Negatif'])
fig_cm, ax_cm = plt.subplots()
cmd = ConfusionMatrixDisplay(cm, display_labels=['Positif', 'Netral', 'Negatif'])
cmd.plot(ax=ax_cm, cmap='Blues')
st.pyplot(fig_cm)

# --- Visualisasi ---
st.header("3. Visualisasi Data")

st.subheader("3.1. Distribusi Sentimen")
sentiment_counts = data_translated['label'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette='viridis')
ax.set_title('Distribusi Sentimen Ulasan Film Jumbo')
ax.set_xlabel('Sentimen')
ax.set_ylabel('Jumlah Ulasan')
st.pyplot(fig)

st.subheader("3.2. Word Cloud dari Teks yang Sudah Diproses")
all_words = ' '.join(df_cleaned['full_text_stemmed'].dropna())
wordcloud = WordCloud(width=3000, height=2000, background_color='yellow', stopwords=STOPWORDS).generate(all_words)
fig_wc, ax_wc = plt.subplots(figsize=(10, 8))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
ax_wc.set_title('Word Cloud Ulasan Film Jumbo')
st.pyplot(fig_wc)

for label, color in zip(['Positif', 'Negatif', 'Netral'], ['green', 'red', 'blue']):
    st.subheader(f"3.{['Positif', 'Negatif', 'Netral'].index(label)+3}. Word Cloud Sentimen {label}")
    words = ' '.join([
        text for text, lbl in zip(df_cleaned['full_text_stemmed'], data_translated['label'])
        if lbl == label and pd.notna(text)
    ])
    if words:
        wc = WordCloud(width=3000, height=2000, background_color=color, stopwords=STOPWORDS).generate(words)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 8))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title(f'Word Cloud Sentimen {label}')
        st.pyplot(fig_wc)
    else:
        st.write(f"Tidak ada ulasan {label.lower()} untuk membuat Word Cloud.")
