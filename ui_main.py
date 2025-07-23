# ui_main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from analiz_motoru import analiz_et, varlik_etiketlerini_getir
from rapor_olusturucu import rapor_metni_olustur, kaydet_rapor
import webbrowser

secilen_cikti_format = "txt"
cikti_klasoru = "C:/Users/Sami/Desktop/chat_gpt/raporlar"
logo_yolu = "C:/Users/Sami/Desktop/chat_gpt/logo.png"

firma_pdf_yolu = ""
kanun_klasoru_yolu = ""

def raporu_ac(dosya_yolu):
    try:
        webbrowser.open(dosya_yolu)
    except Exception as e:
        print(f"Rapor açılırken hata: {e}")

def firma_pdf_sec():
    global firma_pdf_yolu
    yol = filedialog.askopenfilename(filetypes=[("PDF Dosyaları", "*.pdf")])
    if yol:
        firma_pdf_yolu = yol
        log_kutusu.insert(tk.END, f"✅ Firma PDF seçildi: {yol}\n")
        log_kutusu.see(tk.END)

def kanun_klasoru_sec():
    global kanun_klasoru_yolu
    yol = filedialog.askdirectory()
    if yol:
        kanun_klasoru_yolu = yol
        log_kutusu.insert(tk.END, f"✅ Kanun klasörü seçildi: {yol}\n")
        log_kutusu.see(tk.END)

def cikti_format_sec(event=None):
    global secilen_cikti_format
    secilen_cikti_format = cikti_combobox.get()
    log_kutusu.insert(tk.END, f"💾 Çıktı formatı seçildi: {secilen_cikti_format}\n")
    log_kutusu.see(tk.END)

def analiz_baslat():
    if not firma_pdf_yolu or not kanun_klasoru_yolu:
        messagebox.showerror("Eksik Bilgi", "Lütfen firma PDF ve kanun klasörünü seçin.")
        return

    log_kutusu.insert(tk.END, "🚀 Analiz başlatılıyor...\n")
    log_kutusu.update()
    log_kutusu.see(tk.END)

    try:
        en_iyi_metin, en_iyi_skor = analiz_et(firma_pdf_yolu, kanun_klasoru_yolu)
        etiketler = varlik_etiketlerini_getir(firma_pdf_yolu)

        rapor = rapor_metni_olustur(firma_pdf_yolu, en_iyi_metin, en_iyi_skor, etiketler)
        kayit_yolu = kaydet_rapor(rapor, cikti_klasoru, format=secilen_cikti_format)

        log_kutusu.insert(tk.END, f"✅ Rapor oluşturuldu: {kayit_yolu}\n")
        log_kutusu.insert(tk.END, f"🎯 Benzerlik Skoru: {round(en_iyi_skor, 2)}\n")
        log_kutusu.insert(tk.END, f"🧠 Varlık Etiketleri: {', '.join(etiketler)}\n")
        log_kutusu.insert(tk.END, f"📁 Format: {secilen_cikti_format.upper()}\n")
        log_kutusu.see(tk.END)

        if messagebox.askyesno("Analiz Tamamlandı", "Rapor oluşturuldu. Hemen açmak ister misiniz?"):
            raporu_ac(kayit_yolu)

    except Exception as e:
        log_kutusu.insert(tk.END, f"❌ Hata: {str(e)}\n")
        log_kutusu.see(tk.END)

# Arayüz kurulumu
pencere = tk.Tk()
pencere.title("Gabrela Yapay Zeka Denetim Asistanı")
pencere.geometry("960x660")

baslik = tk.Label(pencere, text="Gabrela Yapay Zeka Denetim Asistanı", font=("Arial", 18, "bold"), fg="navy")
baslik.pack(pady=10)

buton_cerceve = tk.Frame(pencere)
buton_cerceve.pack(pady=10)

tk.Button(buton_cerceve, text="1️⃣ Firma PDF Seç", command=firma_pdf_sec).grid(row=0, column=0, padx=10)
tk.Button(buton_cerceve, text="2️⃣ Kanun Klasörü Seç", command=kanun_klasoru_sec).grid(row=0, column=1, padx=10)

cikti_combobox = ttk.Combobox(buton_cerceve, values=["txt", "pdf", "html", "csv"])
cikti_combobox.set("txt")
cikti_combobox.bind("<<ComboboxSelected>>", cikti_format_sec)
cikti_combobox.grid(row=0, column=2, padx=10)

log_kutusu = tk.Text(pencere, height=22, width=112)
log_kutusu.pack(pady=10)

tk.Button(pencere, text="🚀 Analizi Başlat", font=("Arial", 14), bg="green", fg="white", command=analiz_baslat).pack(pady=20)

pencere.mainloop()
