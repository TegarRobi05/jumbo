import streamlit as st
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob
import nltk
from translate import Translator
import preprocessor as p
from wordcloud import WordCloud, STOPWORDS

# Download NLTK data (if not already downloaded in the environment)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('film_jumbo.csv')
    return df

# --- 2. Cleaning Data (from Colab) ---
def clean_text(text):
    text = str(text) # Ensure text is string
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text

# Normalization dictionary (from Colab)
norm = {
    " yg ": " yang ", " gk ": " tidak ", " ga ": " tidak ", " knp ": " kenapa ",
    " ngga ": " tidak ", " gak ": " tidak ", " engga ": " tidak ", " enggak ": " tidak ",
    " nggak ": " tidak ", " enda ": " tidak ", " gua ": " aku ", " gue ": " aku ",
    " gwe ": " aku ", " melek ": " sadar ", " mantap ": " keren ", " drpd ": " daripada ",
    " elu ": " kamu ", " lu ": " kamu ", " lo ": " kamu ", " elo ": " kamu ",
    " nobar ": " nonton bersama ", " krn ": " karena ", " gw ": " aku ", " guwe ": " aku ",
    " ges ": " guys ", " gaes ": " guys ", " kayak ": " seperti ", " skrg ": " sekarang ",
    " taun ": " tahun ", " thh ": " tahun ", " th ": " tahun ", " org ": " orang ",
    " udah ": " sudah ", " kpd ": " kepada ", " gaakan ": " tidak akan ", " udh ": " sudah ",
    " malem ": " malam ", " males ": " malas", " asu ": " anjing ", " dg ": " dengan ",
    " dgn ": " dengan ", " kyk ": " seperti ", " kayaknya ": " sepertinya ", " kyaknya ": " sepertinya ",
    " paslon ": " pasangan calon ", " gaa ": " tidak ", " emg ": " emang ", " asep ": " asap ",
    " bgt ": " banget ", " karna ": " karena ", " muuuanis ": " manis ", " pilem ": " film ",
    " lom ": " belum ", " lbh ": " lebih ", " boring ": " bosan ", " bgttttt ": " banget ",
    " abis ": " habis ", " cuan ": " duit ", " jnck ": " jancok ", " jancuk ": " jancok ",
    " cok ": " jancok ", " jd ": " jadi ", " meleduk ": " meledak ", " kgt ": " kaget ",
    " dpt ": " dapat ", " rmhnya ": " rumahnya ", " rmh ": " rumah ", " nntn ": " nonton ",
    " gla ": " gula ", " byk ": " banyak ", " bnyk ": " banyak ", " kmrn ": " kemaren ",
    " kemarn ": " kemaren ", " kmaren ": " kemaren ", " gpp ": " tidak apa apa",
    " gapapa ": "  tidak apa apa ", " uda ": " sudah ", " udh ": " sudah ", " blm ": " belum ",
    " tp ": " tapi ", " gr ": " gara ", " grgr ": " gara gara ", " kocak ": " lucu ",
    " b aja ": " biasa aja ", " b aj ": "  biasa aja ", " gaperlu ": " tidak perlu ",
    " klean ": " kalean ", " aja ": " saja ", " gitu ": " seperti itu ", " nih ": " ini ",
    " tuh ": " itu ", " dmna ": " dimana ", " kyk gitu ": " seperti itu ", " kyk nya ": " sepertinya ",
    " apa gitu ": " apa seperti itu ", " ngapain ": " mengapa ", " nntn ": " nonton ",
    " bs ": " bisa ", " gaes ": " teman-teman ", " trus ": " terus ", " sdh ": " sudah ",
    " dr ": " dari ", " hrs ": " harus ", " misal ": " misalnya ", " mksd ": " maksud ",
    " plg ": " pulang ", " lg ": " lagi ", " gk ": " tidak ", " g ": " tidak ",
    " dah ": " sudah ", " dalem ": " dalam ", " kalo ": " jika ", " trs ": " terus ",
    " ortu ": " orang tua ", " anak2 ": " anak-anak ", " skr ": " sekarang ", " jd ": " jadi ",
    " dgn ": " dengan ", " mgkn ": " mungkin ", " ngaruh ": " berpengaruh ", " skli ": " sekali ",
    " cm ": " cuma ", " gausah ": " tidak usah ", " begtu ": " begitu ", " bnyk bgt ": " sangat banyak ",
    " btw ": " omong-omong ", " apalagi ": " terlebih lagi ", " tpi ": " tapi ",
    " pdhl ": " padahal ", " kyknya ": " sepertinya ", " soalnya ": " karena ", " jg ": " juga ",
    " kmu ": " kamu ", " aku ": " saya ", " ngerasa ": " merasa ", " kagak ": " tidak ",
    " jadiin ": " jadikan ", " gaje ": " gak jelas ",
}

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

# Stopword removal (from Colab)
@st.cache_resource
def get_stopword_remover():
    more_stop_words = ['tidak']
    stop_words = StopWordRemoverFactory().get_stop_words()
    stop_words.extend(more_stop_words)
    new_array = ArrayDictionary(stop_words)
    return StopWordRemover(new_array)

stop_words_remover_new = get_stopword_remover()

def stopword(str_text):
    return stop_words_remover_new.remove(str_text)

# Stemming (from Colab)
@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = get_stemmer()

def stemming_text(text_cleaning):
    do = []
    for w in text_cleaning.split(): # text_cleaning is already a string after stopword removal
        dt = stemmer.stem(w)
        do.append(dt)
    return " ".join(do)

# Translation (from Colab)
# WARNING: This function can be very slow and might hit API limits if used extensively.
# For a real-world app, consider pre-translating the dataset or using a more robust translation service.
@st.cache_data
def translate_text(text):
    try:
        translator = Translator(from_lang='id', to_lang='en')
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.warning(f"Translation failed for: '{text}'. Error: {e}. Returning original text.")
        return text # Return original text if translation fails

# --- 3. Sentiment Analysis (from Colab) ---
@st.cache_data
def perform_sentiment_analysis(df_processed):
    data_tweet = list(df_processed['english_tweet'])
    status = []
    total_positif = total_negatif = total_netral = 0

    for tweet in data_tweet:
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0.0:
            total_positif += 1
            status.append('Positif')
        elif analysis.sentiment.polarity == 0.0:
            total_netral += 1
            status.append('Netral')
        else:
            total_negatif += 1
            status.append('Negatif')
    
    df_processed['label'] = status
    return df_processed, total_positif, total_netral, total_negatif

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Analisis Sentimen Film Jumbo")

st.title("🎬 Analisis Sentimen Ulasan Film Jumbo")
st.markdown("Aplikasi ini melakukan analisis sentimen pada ulasan film 'Jumbo' dari data yang telah diproses.")

# Load initial data
df_raw = load_data()

st.header("1. Data Awal")
st.write("Berikut adalah 5 baris pertama dari data mentah:")
st.dataframe(df_raw.head())
st.write(f"Total data awal: {df_raw.shape[0]} baris")

# --- Data Preprocessing Steps ---
st.header("2. Pra-pemrosesan Data")

with st.spinner("Melakukan pembersihan teks..."):
    df_processed = df_raw.copy()
    df_processed['full_text'] = df_processed['full_text'].apply(clean_text)
    df_processed['full_text'] = df_processed['full_text'].str.lower()
    st.success("Pembersihan teks selesai (menghapus URL, mention, hashtag, tanda baca, angka, dan mengubah ke huruf kecil).")
    st.dataframe(df_processed.head())

with st.spinner("Melakukan normalisasi teks..."):
    df_processed['full_text'] = df_processed['full_text'].apply(lambda x: normalisasi(x))
    st.success("Normalisasi teks selesai (mengubah singkatan/typo).")
    st.dataframe(df_processed.head())

with st.spinner("Melakukan penghapusan Stopwords..."):
    df_processed['full_text'] = df_processed['full_text'].apply(lambda x: stopword(x))
    st.success("Penghapusan Stopwords selesai.")
    st.dataframe(df_processed.head())

with st.spinner("Melakukan Stemming..."):
    # Tokenize before stemming as per your Colab code's stemming function
    # However, the stemming function in Colab expects a list of words, but the apply function passes a string.
    # I've adjusted the stemming_text function to split the string into words.
    df_processed['full_text'] = df_processed['full_text'].apply(lambda x: stemming_text(x))
    st.success("Stemming selesai.")
    st.dataframe(df_processed.head())

# --- Translation Step ---
st.header("3. Terjemahan ke Bahasa Inggris")
st.warning("Proses terjemahan bisa memakan waktu sangat lama dan mungkin gagal untuk data yang besar. Untuk demo ini, kami akan memuat data yang sudah diterjemahkan dari `StemmingJumbo(2).csv` dan `translateJumboo.csv`.")

# Load the pre-translated data as per your Colab notebook's final steps
@st.cache_data
def load_translated_data():
    # Assuming StemmingJumbo(2).csv is the result of stemming
    df_stemmed = pd.read_csv("StemmingJumbo(2).csv", encoding='latin1')
    
    # Assuming translateJumboo.csv contains the 'english_tweet' column
    # and has been pre-processed to 605 rows as seen in your output
    try:
        df_translated = pd.read_csv("translateJumboo.csv", index_col=0)
        # Merge or ensure consistency if needed. For simplicity, we'll use the 'english_tweet' from this file.
        # Ensure the 'full_text' column is also consistent if you want to display it.
        df_final = df_translated[['full_text', 'english_tweet']].copy()
    except FileNotFoundError:
        st.error("File 'translateJumboo.csv' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if critical file is missing
    
    return df_final

df_final = load_translated_data()
st.write("Data setelah terjemahan (memuat dari file yang sudah ada):")
st.dataframe(df_final.head())
st.write(f"Total data setelah terjemahan: {df_final.shape[0]} baris")


# --- 4. Analisis Sentimen ---
st.header("4. Analisis Sentimen")

df_sentiment, pos_count, neu_count, neg_count = perform_sentiment_analysis(df_final.copy())

st.write("Hasil analisis sentimen:")
st.dataframe(df_sentiment.head())

st.subheader("Ringkasan Sentimen")
col1, col2, col3 = st.columns(3)
col1.metric("Positif", pos_count)
col2.metric("Netral", neu_count)
col3.metric("Negatif", neg_count)

# Plotting sentiment distribution
st.subheader("Distribusi Sentimen")
fig, ax = plt.subplots()
sentiment_counts = df_sentiment['label'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette='viridis')
ax.set_title('Distribusi Sentimen Ulasan Film Jumbo')
ax.set_xlabel('Sentimen')
ax.set_ylabel('Jumlah Ulasan')
st.pyplot(fig)

# --- 5. Visualisasi Word Cloud ---
st.header("5. Visualisasi Word Cloud")

st.subheader("Word Cloud dari Teks Asli (setelah preprocessing)")
all_words_id = ' '.join([text for text in df_sentiment['full_text']])
wordcloud_id = WordCloud(
    width=3000,
    height=2000,
    random_state=3,
    background_color='yellow',
    collocations=False,
    stopwords=STOPWORDS
).generate(all_words_id)

fig_wc_id, ax_wc_id = plt.subplots(figsize=(10, 8))
ax_wc_id.imshow(wordcloud_id, interpolation='bilinear')
ax_wc_id.axis('off')
st.pyplot(fig_wc_id)

st.subheader("Word Cloud dari Teks Terjemahan (English)")
all_words_en = ' '.join([text for text in df_sentiment['english_tweet']])
wordcloud_en = WordCloud(
    width=3000,
    height=2000,
    random_state=3,
    background_color='yellow',
    collocations=False,
    stopwords=STOPWORDS
).generate(all_words_en)

fig_wc_en, ax_wc_en = plt.subplots(figsize=(10, 8))
ax_wc_en.imshow(wordcloud_en, interpolation='bilinear')
ax_wc_en.axis('off')
st.pyplot(fig_wc_en)

st.markdown("---")
st.markdown("Aplikasi ini dibuat berdasarkan analisis sentimen ulasan film 'Jumbo'.")

