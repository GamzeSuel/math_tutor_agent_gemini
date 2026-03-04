Math Tutor Agent (Ollama Local)

Bu proje, Streamlit üzerinden çalışan bir Türkçe matematik öğretmen ajanıdır.
Öğrenciye konu anlatımı sunar, mini quiz uygular, cevapları değerlendirir ve takip/follow-up önerileri üretir.
Proje iki kişi tarafından geliştirilmiştir ve hem eğitim hem de deneysel amaçlarla hazırlanmıştır.

Özellikler

* Ders üretme: Konu açıklaması, bağlantılar, sezgi/intuition, örnekler ve mini quiz

* Cevap değerlendirme: Rubriğe göre puanlama, eksik/hatalı noktaların tespiti, geri bildirim

* Follow-up: Öğrencinin eksik noktalarına yönelik kısa tekrar, örnek ve yeni quiz

* JSON tabanlı şemalar: Pydantic ile yapılandırılmış ve doğrulanmış çıktılar

* Streamlit arayüzü: Kullanıcı dostu web tabanlı deneyim
  
 ![WhatsApp Image 2026-02-26 at 23 06 58](https://github.com/user-attachments/assets/bf41dce4-b25a-4e53-9221-a6436d7c33f2)

  ![WhatsApp Image 2026-02-26 at 23 07 22](https://github.com/user-attachments/assets/aaf36c60-0f1a-484a-a945-a2421fd86a55)
  
  ![WhatsApp Image 2026-02-26 at 23 07 47](https://github.com/user-attachments/assets/594687be-1e0a-41d5-ba74-f038e891a9ac)
  
  ![WhatsApp Image 2026-02-26 at 23 08 06](https://github.com/user-attachments/assets/bd48697f-a00e-4172-aa97-459565ab8e59)

Kurulum!

Python 3.10+ gereklidir.

Gerekli kütüphaneler:

pip install streamlit ollama pydantic python-dotenv json-repair

Opsiyonel: .env dosyası ile OLLAMA_MODEL ve OLLAMA_HOST ayarlanabilir.

.env Dosyası Kullanımı

Projede Ollama LLM modeli ve host adresi ortam değişkenleri ile yönetilebilir.
Bu sayede kod içinde sabit değerler yerine güvenli ve esnek ayarlar kullanılır.

Örnek .env Dosyası

OLLAMA_MODEL=gemma3:1b
OLLAMA_HOST=http://localhost:11434

* OLLAMA_MODEL → kullanılacak LLM modeli

* OLLAMA_HOST → Ollama server host adresi (opsiyonel, yoksa default localhost)

Kod İçinde Kullanımı

from dotenv import load_dotenv
import os

load_dotenv()  # .env içindeki değerleri okur

model = os.getenv("OLLAMA_MODEL") or "gemma3:1b"
host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"

Bu yöntem hem güvenli hem de farklı ortamlar için esneklik sağlar.

Kullanım

Streamlit arayüzünü başlatmak için:

streamlit run app.py

Tarayıcı açıldığında:

1. Konu girilir (örn: logaritma, türev)

2. Seviye seçilir (middle_school / high_school / college)

3. “Ders Üret” butonuna basılır → açıklama, bağlantılar, sezgi ve örnekler görüntülenir

4. Mini quiz çözülür → cevap girilir → “Cevabı Değerlendir” ile skor, eksik/hatalı noktalar ve geri bildirim gösterilir

5. Follow-up bölümü (öğrenci eksik yaptıysa) → kısa tekrar, yeni örnek ve quiz önerisi gösterilir
   
Yapılandırılmış Şemalar

* LessonOutput – Ders anlatımı ve mini quiz bilgisi

* EvalOutput – Öğrenci cevabı değerlendirmesi: skor, eksik/hatalı noktalar, geri bildirim, sonraki adım

* FollowUpOutput – Follow-up soruları, mikro tekrar, yeni örnek ve mini quiz önerisi

Konfigürasyon

* Model seçimi:
agent = OllamaTutorAgent(model="gemma3:1b")

Varsayılan model:

* Ortam değişkeni OLLAMA_MODEL

* Yoksa "gemma3:1b" kullanılır

* Host ayarı (opsiyonel):

agent = OllamaTutorAgent(host="http://localhost:11434")

Önemli Notlar

* Ollama LLM local ortamda çalıştırılmalıdır

* JSON çıktıları Pydantic ile doğrulanır; eksik/hatalı alanlar script tarafından düzeltilir

* Tüm ders ve mini quiz işlemleri Streamlit arayüzünden interaktif şekilde yapılır

Geliştirme Önerileri

* Daha fazla seviye ve konu ekleme

* Takip loglarını kaydedip öğrenci ilerlemesini analiz etme

* Web arayüzünü geliştirme ve daha görselleştirilmiş içerik sunma

## Contributors
### 👥 Proje Katılımcıları

| [<img src="https://github.com/Zeynep35.png" width="100px;"/><br /><sub><b>Zeynep KOÇ</b></sub>](https://github.com/Zeynep35) | [<img src="https://github.com/GamzeSuel.png" width="100px;"/><br /><sub><b>Gamze Süel</b></sub>](https://github.com/GamzeSuel) |
| :---: | :---: |
