import streamlit as st
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

# Download NLTK punkt tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Configuration ---
GITHUB_RAW_URL = "https://raw.githubusercontent.com/your_username/your_repo/main/" # Ganti dengan username dan nama repo Anda
FILM_JUMBO_CSV = GITHUB_RAW_URL + "film_jumbo.csv"
STEMMING_JUMBO_CSV = GITHUB_RAW_URL + "StemmingJumbo(2).csv" # Asumsi file ini sudah ada di repo Anda
TRANSLATE_JUMBO_CSV = GITHUB_RAW_URL + "translateJumboo.csv" # Asumsi file ini sudah ada di repo Anda

# --- Helper Functions (from your Colab notebook) ---

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
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
    " udh ": " udah ", " malem ": " malam ", " males ": " malas", " asu ": " anjing ",
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
    factory = StopWordRemoverFactory()
    more_stop_words = ['tidak']
    stop_words = factory.get_stop_words()
    stop_words.extend(more_stop_words)
    new_array = ArrayDictionary(stop_words)
    stop_words_remover_new = StopWordRemover(new_array)
    return stop_words_remover_new.remove(str_text)

def stemming(text_cleaning):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text_cleaning.split(): # Assuming text_cleaning is a string
        dt = stemmer.stem(w)
        do.append(dt)
    return " ".join(do)

def get_sentiment_label(tweet_text):
    analysis = TextBlob(tweet_text)
    if analysis.sentiment.polarity > 0.0:
        return 'Positif'
    elif analysis.sentiment.polarity == 0.0:
        return 'Netral'
    else:
        return 'Negatif'

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Analisis Sentimen Film Jumbo")

st.title("🎬 Analisis Sentimen Film Jumbo")
st.markdown("Aplikasi ini melakukan analisis sentimen terhadap ulasan film 'Jumbo' menggunakan berbagai teknik pra-pemrosesan teks dan klasifikasi Naive Bayes.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigasi")
menu_selection = st.sidebar.radio(
    "Pilih Menu:",
    ("Beranda", "Pra-pemrosesan Data", "Pelabelan Sentimen", "Klasifikasi Naive Bayes", "Evaluasi Model")
)

# --- Load Data ---
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding='latin1')
    return df

# --- Beranda ---
if menu_selection == "Beranda":
    st.header("Selamat Datang!")
    st.write("Gunakan menu di sidebar untuk menjelajahi proses analisis sentimen.")
    st.subheader("Data Mentah")
    df_raw = load_data(FILM_JUMBO_CSV)
    st.dataframe(df_raw.head())
    st.write(f"Jumlah data mentah: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

# --- Pra-pemrosesan Data ---
elif menu_selection == "Pra-pemrosesan Data":
    st.header("⚙️ Pra-pemrosesan Data")
    st.write("Langkah-langkah untuk membersihkan dan menyiapkan data teks.")

    df_processed = load_data(FILM_JUMBO_CSV).copy()
    df_processed = df_processed[['full_text']]

    preprocessing_step = st.selectbox(
        "Pilih Langkah Pra-pemrosesan:",
        ("Data Awal", "Cleaning", "Normalisasi", "Stopword Removal", "Tokenisasi", "Stemming")
    )

    if preprocessing_step == "Data Awal":
        st.subheader("Data Awal (Setelah Pemilihan Kolom)")
        st.dataframe(df_processed.head())
        st.write(f"Jumlah data: {df_processed.shape[0]} baris, {df_processed.shape[1]} kolom")
        st.write(f"Jumlah duplikat: {df_processed.duplicated().sum()}")
        st.write(f"Jumlah nilai null: \n{df_processed.isnull().sum()}")

    elif preprocessing_step == "Cleaning":
        st.subheader("Hasil Cleaning Data")
        df_cleaned = df_processed.copy()
        df_cleaned['full_text'] = df_cleaned['full_text'].apply(clean_text)
        df_cleaned['full_text'] = df_cleaned['full_text'].str.lower()
        st.dataframe(df_cleaned.head())
        st.write("Langkah ini menghapus URL, mention (@), hashtag (#), tanda baca, angka, dan mengubah teks menjadi huruf kecil.")

    elif preprocessing_step == "Normalisasi":
        st.subheader("Hasil Normalisasi Data")
        df_normalized = df_processed.copy()
        df_normalized['full_text'] = df_normalized['full_text'].apply(clean_text)
        df_normalized['full_text'] = df_normalized['full_text'].str.lower()
        df_normalized['full_text'] = df_normalized['full_text'].apply(normalisasi)
        st.dataframe(df_normalized.head())
        st.write("Langkah ini mengubah kata-kata tidak baku menjadi baku (misal: 'yg' menjadi 'yang').")

    elif preprocessing_step == "Stopword Removal":
        st.subheader("Hasil Stopword Removal")
        df_stopword = df_processed.copy()
        df_stopword['full_text'] = df_stopword['full_text'].apply(clean_text)
        df_stopword['full_text'] = df_stopword['full_text'].str.lower()
        df_stopword['full_text'] = df_stopword['full_text'].apply(normalisasi)
        df_stopword['full_text'] = df_stopword['full_text'].apply(stopword)
        st.dataframe(df_stopword.head())
        st.write("Langkah ini menghapus kata-kata umum yang tidak memiliki makna sentimen (stop words).")

    elif preprocessing_step == "Tokenisasi":
        st.subheader("Hasil Tokenisasi")
        df_tokenized = df_processed.copy()
        df_tokenized['full_text'] = df_tokenized['full_text'].apply(clean_text)
        df_tokenized['full_text'] = df_tokenized['full_text'].str.lower()
        df_tokenized['full_text'] = df_tokenized['full_text'].apply(normalisasi)
        df_tokenized['full_text'] = df_tokenized['full_text'].apply(stopword)
        df_tokenized['tokenized_text'] = df_tokenized['full_text'].apply(lambda x: x.split())
        st.dataframe(df_tokenized[['full_text', 'tokenized_text']].head())
        st.write("Langkah ini memecah teks menjadi unit-unit kata (token).")

    elif preprocessing_step == "Stemming":
        st.subheader("Hasil Stemming")
        # Load data after stemming from GitHub as per your Colab output
        df_stemmed = load_data(STEMMING_JUMBO_CSV)
        st.dataframe(df_stemmed.head())
        st.write("Langkah ini mengubah kata berimbuhan menjadi kata dasar.")
        st.write("Catatan: Proses stemming memakan waktu. Data yang ditampilkan di sini dimuat dari file CSV yang sudah distemming.")

# --- Pelabelan Sentimen ---
elif menu_selection == "Pelabelan Sentimen":
    st.header("🏷️ Pelabelan Sentimen")
    st.write("Melakukan pelabelan sentimen (Positif, Netral, Negatif) pada data yang sudah diproses.")

    # Load data after translation from GitHub as per your Colab output
    df_labeled = load_data(TRANSLATE_JUMBO_CSV)
    
    st.subheader("Data Setelah Translasi dan Pelabelan")
    st.dataframe(df_labeled.head())

    st.subheader("Distribusi Sentimen")
    sentiment_counts = df_labeled['label'].value_counts()
    st.bar_chart(sentiment_counts)

    st.write(f"**Jumlah Sentimen:**")
    st.write(f"- Positif: {sentiment_counts.get('Positif', 0)}")
    st.write(f"- Netral: {sentiment_counts.get('Netral', 0)}")
    st.write(f"- Negatif: {sentiment_counts.get('Negatif', 0)}")
    st.write(f"Total Data: {len(df_labeled)}")

# --- Klasifikasi Naive Bayes ---
elif menu_selection == "Klasifikasi Naive Bayes":
    st.header("📊 Klasifikasi Naive Bayes")
    st.write("Melatih model Naive Bayes untuk klasifikasi sentimen.")

    df_model = load_data(TRANSLATE_JUMBO_CSV)
    
    if 'english_tweet' not in df_model.columns or 'label' not in df_model.columns:
        st.error("Kolom 'english_tweet' atau 'label' tidak ditemukan. Pastikan file `translateJumboo.csv` sudah benar.")
    else:
        X = df_model['english_tweet']
        y = df_model['label']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Pembagian Data")
        st.write(f"Jumlah data training: {len(X_train)}")
        st.write(f"Jumlah data testing: {len(X_test)}")

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        st.subheader("TF-IDF Vectorization")
        st.write("Teks diubah menjadi representasi numerik menggunakan TF-IDF.")
        st.write(f"Dimensi data training setelah TF-IDF: {X_train_tfidf.shape}")
        st.write(f"Dimensi data testing setelah TF-IDF: {X_test_tfidf.shape}")

        # Train Naive Bayes Model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        st.success("Model Naive Bayes berhasil dilatih!")

        # Store model and vectorizer in session state for later use
        st.session_state['model'] = model
        st.session_state['vectorizer'] = vectorizer
        st.session_state['X_test_tfidf'] = X_test_tfidf
        st.session_state['y_test'] = y_test

        st.subheader("Prediksi Sentimen (Contoh)")
        sample_text = st.text_area("Masukkan teks untuk diprediksi:", "This movie is amazing and I love it!")
        if st.button("Prediksi"):
            if 'vectorizer' in st.session_state:
                sample_text_tfidf = st.session_state['vectorizer'].transform([sample_text])
                prediction = st.session_state['model'].predict(sample_text_tfidf)
                st.write(f"Sentimen diprediksi: **{prediction[0]}**")
            else:
                st.warning("Model belum dilatih. Silakan kunjungi menu 'Klasifikasi Naive Bayes' terlebih dahulu.")

# --- Evaluasi Model ---
elif menu_selection == "Evaluasi Model":
    st.header("📈 Evaluasi Model")
    st.write("Mengevaluasi performa model Naive Bayes.")

    if 'model' in st.session_state and 'X_test_tfidf' in st.session_state and 'y_test' in st.session_state:
        model = st.session_state['model']
        X_test_tfidf = st.session_state['X_test_tfidf']
        y_test = st.session_state['y_test']

        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader("Metrik Evaluasi")
        st.write(f"**Akurasi:** {accuracy:.4f}")
        st.write(f"**Presisi:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1-Score:** {f1:.4f}")

        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Netral', 'Negatif'])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positif', 'Netral', 'Negatif'], yticklabels=['Positif', 'Netral', 'Negatif'], ax=ax)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    else:
        st.warning("Model belum dilatih. Silakan kunjungi menu 'Klasifikasi Naive Bayes' terlebih dahulu.")

