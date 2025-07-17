import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ==================== Bagian 1: Respon Chat Umum ====================
responses = {
    "hai": ["Hai juga!", "Halo!", "Hai, ada yang bisa aku bantu?"],
    "mau": ["Mari kita lanjut"],
    "yee": ["Bacot pantek"],
    "aku tidak baik baik saja": ["Bekerjalah lebih keras dasar pemalas"],
    "bukan": ["Lalu apa yang bisa aku bantu?"],
    "hallo": ["Hai juga!", "Halo!", "Hai, ada yang bisa aku bantu?"],
    "halo": ["Hai juga!", "Halo!", "Hai, ada yang bisa aku bantu?"],
    "boleh" : ["ayo kamu ingin melakukan apa?"],
    "apa kabar": ["Aku baik-baik saja, kamu?", "Luar biasa! Kamu gimana?", "Senang ngobrol denganmu!"],
    "baik": ["senang mendengarnya", "aku senang kamu juga merasa baik", "apa ada yang bisa aku bantu?"],
    "siapa kamu": ["Aku adalah chatbot konseling buatan Python.", "Aku AI kecil yang sedang belajar memahami siswa."],
    "kamu siapa": ["Aku adalah chatbot konseling buatan Python.", "Aku AI kecil yang sedang belajar memahami siswa."],
    "siapa namamu": ["Aku adalah chatbot konseling buatan Python.", "Aku AI kecil yang sedang belajar memahami siswa."],
    "nama kamu siapa": ["Aku adalah chatbot konseling buatan Python.", "Aku AI kecil yang sedang belajar memahami siswa."],
    "nama kamu": ["Aku adalah chatbot konseling buatan Python.", "Aku AI kecil yang sedang belajar memahami siswa."],
    "terima kasih": ["Sama-sama!", "Dengan senang hati!", "Kapan saja!"],
    "aku malas" : ["hai aku bisa membantumu hari ini" , "ayo lakukan sesuatu" , "mau coba hal lain?"],
    "masih malas" :["mau coba hal lain?"],
    "melakukan apa": ["kamu bisa mengajukan pertanyaan", "beraktivitas di luar ruangan", "kamu bisa beristirahat sejenak"],
    "ya": ["baiklah semoga harimu menyenangkan", "nice!!", "kamu ingin aku melakukan apa"],
    "keluar": ["Untuk keluar ketik 'exit'."],
    "default": ["Maaf, aku belum mengerti itu. Coba ketik keluhan atau ketik 'menu' Untuk keluar ketik 'exit'."]
}

def chatbot_response(user_input):
    user_input = user_input.lower()
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
    return random.choice(responses["default"])

# ==================== Bagian 2: Sistem Pakar ====================
def inferensi(nilai, kehadiran, minat):
    rekomendasi = []
    if nilai < 60 and minat == "rendah":
        rekomendasi.append("Konseling akademik intensif.")
    elif nilai < 70:
        rekomendasi.append("Pendampingan belajar rutin.")
    if kehadiran < 75:
        rekomendasi.append("Konseling kehadiran dan motivasi.")
    if minat == "tinggi" and nilai >= 70 and kehadiran >= 75:
        rekomendasi.append("Tidak perlu konseling saat ini. Pertahankan!")
    if not rekomendasi:
        rekomendasi.append("Perlu observasi lanjutan oleh guru BK.")
    return rekomendasi

# ==================== Bagian 3: AI NLP (Text Classification) ====================
keluhan = [
    "saya sering tidak masuk sekolah",
    "saya tidak paham pelajaran matematika",
    "saya merasa tidak semangat belajar",
    "saya suka belajar dan tidak punya masalah",
    "saya cemas dan tidak fokus saat ujian",
    "saya malas belajar dan sering bolos",
    "saya butuh bantuan memahami pelajaran fisika",
    "saya semangat belajar dan nilai saya bagus"
]

label = [
    "Kamu butuh Konseling kehadiran",
    "Kamu butuh Konseling akademik",
    "Kamu butuh Konseling motivasi",
    "Kamu Tidak perlu konseling",
    "Kamu butuh Konseling psikologis",
    "Kamu butuh Konseling akademik dan kehadiran",
    "Kamu butuh Konseling akademik",
    "Kamu butuh Tidak perlu konseling"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(keluhan)
model = MultinomialNB()
model.fit(X, label)

def prediksi_konseling(teks_masukan):
    vektor_input = vectorizer.transform([teks_masukan])
    hasil = model.predict(vektor_input)
    return hasil[0]

# ==================== Bagian 4: Antarmuka Utama ====================
def main():
    print("👋 Halo! Aku Chatbot Konseling Akademik. Ketik 'menu' untuk lihat opsi, atau langsung ngobrol aja, ketik 'exit' untuk keluar.")
    while True:
        user_input = input("Kamu: ").lower()
        
        if user_input == "exit":
            print("Chatbot: Sampai jumpa, semoga harimu menyenangkan!")
            break
        elif user_input == "menu":
            print("Chatbot: Pilih metode:")
            print("1. Diagnosa Berdasarkan Data Akademik")
            print("2. Konseling Berdasarkan Cerita Siswa (AI)")
            pilihan = input("Pilih (1/2): ")

            
            if pilihan == "1":
                try:
                    nama = input("Nama siswa: ")
                    nilai = float(input("Nilai rata-rata akademik (0–100): "))
                    kehadiran = float(input("Persentase kehadiran (%): "))
                    minat = input("Tingkat minat belajar (rendah/sedang/tinggi): ").lower()
                    hasil = inferensi(nilai, kehadiran, minat)
                    print(f"\nRekomendasi untuk {nama}:")
                    for r in hasil:
                        print("- " + r)
                except ValueError:
                    print("Input tidak valid. Gunakan angka.")
            
            elif pilihan == "2":
                cerita = input("Ceritakan masalah kamu: ")
                hasil = prediksi_konseling(cerita)
                print("Rekomendasi Konseling:", hasil)
            else:
                print("Pilihan tidak valid.")
        
        elif any(kata in user_input for kata in keluhan):
            hasil = prediksi_konseling(user_input)
            print("Chatbot (AI Konseling):", hasil)
        else:
            print("Chatbot:", chatbot_response(user_input))

if __name__ == "__main__":
    main()