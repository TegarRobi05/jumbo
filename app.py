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
import nltk

# Download NLTK punkt tokenizer (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(layout="wide")

st.title("Analisis Sentimen Film Jumbo")
st.write("Aplikasi ini melakukan analisis sentimen terhadap ulasan film 'Jumbo' dari data yang telah diproses.")

# --- Fungsi-fungsi Preprocessing dari Google Colab Anda ---

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text

norm = {
    " yg ": " yang ", " gk ": " tidak ", " ga ": " tidak ", " knp ": " kenapa ",
    " ngga ": " tidak ", " ga ": " tidak ", " gak ": " tidak ", " engga ": " tidak ",
    " enggak ": " tidak ", " nggak ": " tidak ", " enda ": " tidak ", " gua ": " aku ",
    " gue ": " aku ", " gwe ": " aku ", " melek ": " sadar ", " mantap ": " keren ",
    " drpd ": " daripada ", " elu ": " kamu ", " lu ": " kamu ", " lo ": " kamu ",
    " elo ": " kamu ", " nobar ": " nonton bersama ", " krn ": " karena ", " gw ": " aku ",
    " guwe ": " aku ", " ges ": " guys ", " gaes ": " guys ", " kayak ": " seperti ",
    " skrg ": " sekarang ", " taun ": " tahun ", " thh ": " tahun ", " th ": " tahun ",
    " org ": " orang ", " udah ": " sudah ", " kpd ": " kepada ", " gaakan ": " tidak akan ",
    " udh ": " sudah ", " malem ": " malam ", " males ": " malas", " asu ": " anjing ",
    " dg ": " dengan ", " dgn ": " dengan ", " kyk ": " seperti ", " kayaknya ": " sepertinya ",
    " kyaknya ": " sepertinya ", " paslon ": " pasangan calon ", " gaa ": " tidak ",
    " emg ": " emang ", " asep ": " asap ", " bgt ": " banget ", " karna ": " karena ",
    " muuuanis ": " manis ", " pilem ": " film ", " lom ": " belum ", " lbh ": " lebih ",
    " boring ": " bosan ", " bgttttt ": " banget ", " abis ": " habis ", " cuan ": " duit ",
    " jnck ": " jancok ", " jancuk ": " jancok ", " cok ": " jancok ", " jd ": " jadi ",
    " knp ": " kenapa ", " meleduk ": " meledak ", " kgt ": " kaget ", " dpt ": " dapat ",
    " rmhnya ": " rumahnya ", " rmh ": " rumah ", " nntn ": " nonton ", " gla ": " gula ",
    " byk ": " banyak ", " bnyk ": " banyak ", " kmrn ": " kemaren ", " kemarn ": " kemaren ",
    " kmaren ": " kemaren ", " gpp ": " tidak apa apa", " gapapa ": "  tidak apa apa ",
    " uda ": " sudah ", " udh ": " sudah ", " blm ": " belum ", " tp ": " tapi ",
    " gr ": " gara ", " grgr ": " gara gara ", " kocak ": " lucu ", " b aja ": " biasa aja ",
    " b aj ": "  biasa aja ", " gaperlu ": " tidak perlu ", " klean ": " kalean ",
    " aja ": " saja ", " gitu ": " seperti itu ", " nih ": " ini ", " tuh ": " itu ",
    " dmna ": " dimana ", " kyk gitu ": " seperti itu ", " kyk nya ": " sepertinya ",
    " apa gitu ": " apa seperti itu ", " ngapain ": " mengapa ", " nntn ": " nonton ",
    " bs ": " bisa ", " gaes ": " teman-teman ", " trus ": " terus ", " sdh ": " sudah ",
    " dr ": " dari ", " hrs ": " harus ", " misal ": " misalnya ", " mksd ": " maksud ",
    " plg ": " pulang ", " lg ": " lagi ", " gk ": " tidak ", " g ": " tidak ",
    " dah ": " sudah ", " dalem ": " dalam ", " kalo ": " jika ", " trs ": " terus ",
    " ortu ": " orang tua ", " anak2 ": " anak-anak ", " skr ": " sekarang ",
    " jd ": " jadi ", " dgn ": " dengan ", " mgkn ": " mungkin ", " ngaruh ": " berpengaruh ",
    " skli ": " sekali ", " cm ": " cuma ", " gausah ": " tidak usah ", " begtu ": " begitu ",
    " bnyk bgt ": " sangat banyak ", " btw ": " omong-omong ", " apalagi ": " terlebih lagi ",
    " tpi ": " tapi ", " pdhl ": " padahal ", " kyknya ": " sepertinya ", " soalnya ": " karena ",
    " jg ": " juga ", " kmu ": " kamu ", " aku ": " saya ", " ngerasa ": " merasa ",
    " kagak ": " tidak ", " jadiin ": " jadikan ", " gaes ": " teman-teman ", " gaje ": " gak jelas ",
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
    do = []
    for w in text_cleaning.split(): # Split the text into words for stemming
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = " ".join(do)
    return d_clean

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
    df_cleaned = df.drop_duplicates(subset='full_text').copy() # Use .copy() to avoid SettingWithCopyWarning
    st.write(f"Jumlah duplikat yang dihapus: {df.duplicated().sum()}")
    st.write(f"Ukuran data setelah menghapus duplikat: {df_cleaned.shape}")
    st.write("Data awal (5 baris pertama):")
    st.dataframe(df.head())

    st.subheader("1.2. Case Folding, URL, Mention, Hashtag, dan Punctuation Removal")
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(clean_text)
    df_cleaned['full_text'] = df_cleaned['full_text'].str.lower()
    st.write("Data setelah cleaning dan case folding (5 baris pertama):")
    st.dataframe(df_cleaned.head())

    st.subheader("1.3. Normalisasi Kata")
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(lambda x: normalisasi(x))
    st.write("Data setelah normalisasi (5 baris pertama):")
    st.dataframe(df_cleaned.head())

    st.subheader("1.4. Stopword Removal")
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(lambda x: stopword(x))
    st.write("Data setelah stopword removal (5 baris pertama):")
    st.dataframe(df_cleaned.head())

    st.subheader("1.5. Stemming")
    # Apply stemming to the cleaned text
    df_cleaned['full_text_stemmed'] = df_cleaned['full_text'].apply(lambda x: steming(x))
    st.write("Data setelah stemming (5 baris pertama):")
    st.dataframe(df_cleaned.head())

# --- Sentiment Analysis ---
st.header("2. Analisis Sentimen")

# Load pre-translated data (assuming you have translateJumboo.csv from your Colab)
# If you want to translate on the fly, it will be very slow and might hit API limits.
# It's better to use the pre-translated file.
try:
    data_translated = pd.read_csv('translateJumboo.csv', index_col=0)
    # Ensure 'english_tweet' column exists
    if 'english_tweet' not in data_translated.columns:
        st.error("Kolom 'english_tweet' tidak ditemukan di translateJumboo.csv. Pastikan file sudah benar.")
        st.stop()
except FileNotFoundError:
    st.error("File 'translateJumboo.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    st.stop()

# Perform sentiment analysis using TextBlob on the English tweets
data_tweet = list(data_translated['english_tweet'])

status = []
total_positif = 0
total_negatif = 0
total_netral = 0

for tweet in data_tweet:
    analysis = TextBlob(str(tweet)) # Ensure tweet is string
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
st.write(f"Jumlah sentimen Positif: {total_positif}")
st.write(f"Jumlah sentimen Netral: {total_netral}")
st.write(f"Jumlah sentimen Negatif: {total_negatif}")
st.write(f"Total Data: {len(data_translated)}")

st.write("Data dengan label sentimen (5 baris pertama):")
st.dataframe(data_translated.head())

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
all_words = ' '.join([text for text in df_cleaned['full_text_stemmed'].dropna()]) # Use stemmed text for word cloud
wordcloud = WordCloud(
    width=3000,
    height=2000,
    random_state=3,
    background_color='yellow',
    collocations=False,
    stopwords=STOPWORDS
).generate(all_words)

fig_wc, ax_wc = plt.subplots(figsize=(10, 8))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
ax_wc.set_title('Word Cloud Ulasan Film Jumbo (Setelah Preprocessing)')
st.pyplot(fig_wc)

st.subheader("3.3. Word Cloud dari Sentimen Positif")
positive_words = ' '.join([text for text, label in zip(df_cleaned['full_text_stemmed'], data_translated['label']) if label == 'Positif' and pd.notna(text)])
if positive_words:
    wordcloud_pos = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='green',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(positive_words)
    fig_wc_pos, ax_wc_pos = plt.subplots(figsize=(10, 8))
    ax_wc_pos.imshow(wordcloud_pos, interpolation='bilinear')
    ax_wc_pos.axis('off')
    ax_wc_pos.set_title('Word Cloud Sentimen Positif')
    st.pyplot(fig_wc_pos)
else:
    st.write("Tidak ada ulasan positif untuk membuat Word Cloud.")

st.subheader("3.4. Word Cloud dari Sentimen Negatif")
negative_words = ' '.join([text for text, label in zip(df_cleaned['full_text_stemmed'], data_translated['label']) if label == 'Negatif' and pd.notna(text)])
if negative_words:
    wordcloud_neg = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='red',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(negative_words)
    fig_wc_neg, ax_wc_neg = plt.subplots(figsize=(10, 8))
    ax_wc_neg.imshow(wordcloud_neg, interpolation='bilinear')
    ax_wc_neg.axis('off')
    ax_wc_neg.set_title('Word Cloud Sentimen Negatif')
    st.pyplot(fig_wc_neg)
else:
    st.write("Tidak ada ulasan negatif untuk membuat Word Cloud.")

st.subheader("3.5. Word Cloud dari Sentimen Netral")
neutral_words = ' '.join([text for text, label in zip(df_cleaned['full_text_stemmed'], data_translated['label']) if label == 'Netral' and pd.notna(text)])
if neutral_words:
    wordcloud_neu = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='blue',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(neutral_words)
    fig_wc_neu, ax_wc_neu = plt.subplots(figsize=(10, 8))
    ax_wc_neu.imshow(wordcloud_neu, interpolation='bilinear')
    ax_wc_neu.axis('off')
    ax_wc_neu.set_title('Word Cloud Sentimen Netral')
    st.pyplot(fig_wc_neu)
else:
    st.write("Tidak ada ulasan netral untuk membuat Word Cloud.")

