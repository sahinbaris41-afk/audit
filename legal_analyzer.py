import os
# TensorFlow uyarılarını bastırma
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import threading
import time
import json
import csv
import sqlite3
import platform
import subprocess
import webbrowser
from collections import defaultdict
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
import markdown
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdfminer.high_level import extract_text as pdfminer_extract
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import logging

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('legal_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NLTK veri setlerini yerel dizine yükleme
def setup_nltk_data():
    try:
        nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path)
        nltk.data.path.append(nltk_data_path)
        
        # Stopwords ve punkt veri setlerini kontrol et
        try:
            stopwords.words('turkish')
        except LookupError:
            logger.warning("NLTK stopwords indiriliyor...")
            nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        
        try:
            nltk.tokenize.punkt
        except LookupError:
            logger.warning("NLTK punkt indiriliyor...")
            nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
            
        logger.info("NLTK veri setleri hazır")
    except Exception as e:
        logger.error(f"NLTK veri setleri yüklenemedi: {str(e)}")
        messagebox.showwarning("Uyarı", f"NLTK veri setleri yüklenemedi, metin analizi sınırlı olabilir: {str(e)}")

# Langdetect için deterministik çıktı
DetectorFactory.seed = 0

class EnhancedLegalAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gelişmiş Hukuki Metin Analiz ve Denetim Sistemi")
        self.root.option_add('*TCombobox*Listbox.font', ('Arial', 10))  # Yüksek DPI için

        # Ekran boyutlarına uyum
        ekran_genislik = self.root.winfo_screenwidth()
        ekran_yukseklik = self.root.winfo_screenheight()
        pencere_genislik = min(1000, ekran_genislik - 100)
        pencere_yukseklik = min(700, ekran_yukseklik - 100)
        self.root.geometry(f"{pencere_genislik}x{pencere_yukseklik}")
        self.root.resizable(True, True)

        # Değişkenler
        self.logo_path = None
        self.doc_folder = None
        self.law_folder = None
        self.output_file = None
        self.output_format = "txt"
        self.log_records = []
        self.analysis_result = None
        self.cancel_flag = False
        self.db_lock = threading.Lock()  # Veritabanı için kilit
        self.conn = None  # Veritabanı bağlantısı
        self.cursor = None  # Veritabanı imleci

        # Denetim başlıkları ve anahtar kelimeler
        self.audit_titles = [
            "Vergi Usul Kanunu", "Türk Ticaret Kanunu", "İş Kanunu", "KVKK",
            "Bankacılık Kanunu", "Sermaye Piyasası Kanunu", "Çevre Kanunu",
            "Tüketici Hakları Kanunu", "Rekabet Kanunu", "Kamu İhale Kanunu"
        ]
        self.audit_keywords = {
            "Vergi Usul Kanunu": ["vergi", "beyanname", "kdv", "matrah", "muhasebe", "defter", "fatura"],
            "Türk Ticaret Kanunu": ["ticaret", "şirket", "sermaye", "ortak", "ticari defter", "sözleşme"],
            "İş Kanunu": ["çalışan", "maaş", "izin", "sözleşme", "işçi", "işveren", "kıdem"],
            "KVKK": ["kişisel veri", "kvkk", "gizlilik", "güvenlik", "veri koruma"],
            "Bankacılık Kanunu": ["banka", "kredi", "mevduat", "faiz", "düzenleme", "lisans"],
            "Sermaye Piyasası Kanunu": ["hisse", "sermaye", "borsa", "menkul kıymet", "spk"],
            "Çevre Kanunu": ["çevre", "atık", "emisyon", "çevresel etki", "sürdürülebilirlik"],
            "Tüketici Hakları Kanunu": ["tüketici", "hak", "satış", "garanti", "şikayet"],
            "Rekabet Kanunu": ["rekabet", "tekel", "kartel", "piyasa", "cezalar"],
            "Kamu İhale Kanunu": ["ihale", "kamu", "sözleşme", "teklif", "ihale şartnamesi"]
        }

        # Rapor seçenekleri
        self.report_options = {
            "Genel Bilgiler": tk.BooleanVar(value=True),
            "Kelime Frekansları": tk.BooleanVar(value=True),
            "Uyumluluk Durumu": tk.BooleanVar(value=True),
            "Örnek Madde": tk.BooleanVar(value=True),
            "Eksiklik Tespiti": tk.BooleanVar(value=True),
            "Karşılaştırmalı Analiz": tk.BooleanVar(value=True)
        }

        # NLTK veri setlerini yükle
        setup_nltk_data()

        # BERT modeli
        self.bert_model = None
        self.bert_tokenizer = None
        self.load_bert_model()

        # Arayüz ve veritabanı oluşturma
        self.create_ui()
        self.create_database()

        # Klavye kısayolları
        self.root.bind('<F5>', lambda e: self.select_folder("doc"))
        self.root.bind('<F6>', lambda e: self.select_folder("law"))
        self.root.bind('<F7>', lambda e: self.select_output_file())
        self.root.bind('<F8>', lambda e: self.start_analysis())

        logger.info("Uygulama başlatıldı. Lütfen doküman ve kanun klasörlerini seçin.")

    def load_bert_model(self):
        """BERT modelini yükler ve önbellekte tutar."""
        try:
            # Yerel model dizini
            model_path = os.path.join(os.getcwd(), 'bert_model')
            if os.path.exists(model_path):
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                logger.info("BERT modeli yerel dizinden yüklendi")
            else:
                logger.warning("Yerel BERT modeli bulunamadı, Hugging Face'ten indiriliyor...")
                self.bert_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased")
                # Modeli yerel dizine kaydet
                self.bert_tokenizer.save_pretrained(model_path)
                self.bert_model.save_pretrained(model_path)
                logger.info("BERT modeli indirildi ve yerel dizine kaydedildi")
        except Exception as e:
            logger.error(f"BERT modeli yüklenemedi: {str(e)}")
            messagebox.showwarning("Uyarı", "BERT modeli yüklenemedi. Derin analiz devre dışı bırakılacak.")
            self.bert_model = None
            self.bert_tokenizer = None

    def create_database(self):
        """Veritabanını oluşturur ve bağlantıyı kurar."""
        try:
            with self.db_lock:
                # Veritabanı dosyasını çalışma dizininde oluştur
                db_path = os.path.join(os.getcwd(), 'analysis_records.db')
                # Yazma izni kontrolü
                if not os.access(os.getcwd(), os.W_OK):
                    raise PermissionError(f"Yazma izni yok: {os.getcwd()}")
                self.conn = sqlite3.connect(db_path)
                self.cursor = self.conn.cursor()
                self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_date TEXT,
                        doc_folder TEXT,
                        law_folder TEXT,
                        total_docs INTEGER,
                        total_laws INTEGER,
                        total_articles INTEGER,
                        total_chars INTEGER,
                        result_file TEXT
                    )
                ''')
                self.conn.commit()
            logger.info("Veritabanı başlatıldı")
        except (sqlite3.Error, PermissionError) as e:
            logger.error(f"Veritabanı hatası: {str(e)}")
            messagebox.showerror("Hata", f"Veritabanı başlatılamadı: {str(e)}")
            self.conn = None
            self.cursor = None

    def create_ui(self):
        """Kullanıcı arayüzünü oluşturur."""
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Logo ve başlık
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=5)
        self.logo_label = ttk.Label(header_frame, text="Logo Yok", font=("Arial", 10))
        self.logo_label.grid(row=0, column=0, sticky="w", padx=5)
        ttk.Button(header_frame, text="Logo Seç", command=self.select_logo).grid(row=0, column=1, sticky="w", padx=5)

        # Yardım menüsü
        help_button = ttk.Button(header_frame, text="Yardım", command=self.show_help)
        help_button.grid(row=0, column=2, sticky="e", padx=5)
        header_frame.grid_columnconfigure(2, weight=1)

        # Dosya seçim alanı
        file_frame = ttk.LabelFrame(main_frame, text="Dosya Seçimleri")
        file_frame.grid(row=1, column=0, sticky="ew", pady=5)

        # Şirket dokümanları
        ttk.Label(file_frame, text="Şirket Dokümanları Klasörü:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.doc_label = ttk.Label(file_frame, text="Seçilmedi")
        self.doc_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(file_frame, text="Seç (F5)", command=lambda: self.select_folder("doc")).grid(row=0, column=2, padx=5, pady=2)

        # Kanun PDF'leri
        ttk.Label(file_frame, text="Kanun PDF'leri Klasörü:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.law_label = ttk.Label(file_frame, text="Seçilmedi")
        self.law_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(file_frame, text="Seç (F6)", command=lambda: self.select_folder("law")).grid(row=1, column=2, padx=5, pady=2)

        # Çıktı dosyası
        ttk.Label(file_frame, text="Çıktı Dosyası:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.output_label = ttk.Label(file_frame, text="Seçilmedi")
        self.output_label.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(file_frame, text="Seç (F7)", command=self.select_output_file).grid(row=2, column=2, padx=5, pady=2)

        # Format seçimi
        ttk.Label(file_frame, text="Çıktı Formatı:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.format_var = tk.StringVar(value="txt")
        formats_frame = ttk.Frame(file_frame)
        formats_frame.grid(row=3, column=1, columnspan=2, sticky="w")
        for col, fmt in enumerate([("TXT", "txt"), ("JSON", "json"), ("CSV", "csv"), ("HTML", "html")]):
            ttk.Radiobutton(formats_frame, text=fmt[0], variable=self.format_var, value=fmt[1]).grid(row=0, column=col, sticky="w", padx=2)

        # Analiz seçenekleri
        options_frame = ttk.LabelFrame(main_frame, text="Analiz Seçenekleri")
        options_frame.grid(row=2, column=0, sticky="ew", pady=5)

        # Denetim başlığı
        ttk.Label(options_frame, text="Denetim Başlığı:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.audit_var = tk.StringVar(value=self.audit_titles[0])
        audit_menu = ttk.OptionMenu(options_frame, self.audit_var, self.audit_titles[0], *self.audit_titles)
        audit_menu.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # BERT analizi
        self.bert_active = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="BERT ile Derin Analiz", variable=self.bert_active).grid(row=0, column=2, sticky="w", padx=5)

        # Madde aralığı
        ttk.Label(options_frame, text="Madde Aralığı:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        range_frame = ttk.Frame(options_frame)
        range_frame.grid(row=1, column=1, columnspan=2, sticky="w")
        ttk.Label(range_frame, text="Başlangıç:").grid(row=0, column=0, sticky="w")
        self.start_entry = ttk.Entry(range_frame, width=5)
        self.start_entry.grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(range_frame, text="Bitiş:").grid(row=0, column=2, sticky="w", padx=(10, 2))
        self.end_entry = ttk.Entry(range_frame, width=5)
        self.end_entry.grid(row=0, column=3, sticky="w", padx=2)

        # Rapor özelleştirme
        ttk.Label(options_frame, text="Rapor İçeriği:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        report_frame = ttk.Frame(options_frame)
        report_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        for i, (option, var) in enumerate(self.report_options.items()):
            ttk.Checkbutton(report_frame, text=option, variable=var).grid(row=i//3, column=i%3, sticky="w", padx=2)

        # Kontrol butonları
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, sticky="ew", pady=5)
        ttk.Button(control_frame, text="Analiz Başlat (F8)", command=self.start_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="İptal", command=self.cancel_analysis).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Grafik Göster", command=self.show_graph).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Rapor Aç", command=self.open_report).grid(row=0, column=3, padx=5)
        control_frame.grid_columnconfigure(4, weight=1)

        # İlerleme çubuğu
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.grid(row=4, column=0, sticky="ew", pady=5)
        self.status_label = ttk.Label(main_frame, text="Hazır", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=5, column=0, sticky="ew", pady=(0, 5))

        # Sekmeler
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=6, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(6, weight=1)

        # Log sekmesi
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="İşlem Kayıtları")
        self.log_text = ScrolledText(log_frame, font=("Arial", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.log_text.config(state=tk.DISABLED)

        # Önizleme sekmesi
        preview_frame = ttk.Frame(notebook)
        notebook.add(preview_frame, text="İçerik Önizleme")
        self.preview_text = ScrolledText(preview_frame, font=("Arial", 10))
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_text.config(state=tk.DISABLED)

        # Geçmiş analizler
        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="Geçmiş Analizler")
        self.history_tree = ttk.Treeview(history_frame, columns=("ID", "Tarih", "Dokümanlar", "Kanunlar", "Dosya"), show='headings')
        self.history_tree.heading("ID", text="ID")
        self.history_tree.heading("Tarih", text="Tarih")
        self.history_tree.heading("Dokümanlar", text="Doküman Klasörü")
        self.history_tree.heading("Kanunlar", text="Kanun Klasörü")
        self.history_tree.heading("Dosya", text="Sonuç Dosyası")
        self.history_tree.column("ID", width=50)
        self.history_tree.column("Tarih", width=120)
        self.history_tree.column("Dokümanlar", width=200)
        self.history_tree.column("Kanunlar", width=200)
        self.history_tree.column("Dosya", width=200)
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        history_frame.grid_rowconfigure(0, weight=1)
        history_frame.grid_columnconfigure(0, weight=1)
        self.history_tree.bind('<Double-1>', self.open_history_report)

        # Veritabanı bağlantısı başarılıysa geçmişi yükle
        if self.cursor is not None:
            self.load_history()

    def show_help(self):
        """Klavye kısayolları için yardım penceresi gösterir."""
        messagebox.showinfo(
            "Yardım",
            "Klavye Kısayolları:\n"
            "F5: Şirket Dokümanları Klasörü Seç\n"
            "F6: Kanun PDF Klasörü Seç\n"
            "F7: Çıktı Dosyası Seç\n"
            "F8: Analizi Başlat"
        )

    def select_logo(self):
        """Logo dosyasını seçer ve arayüzde gösterir."""
        path = filedialog.askopenfilename(title="Logo Seç", filetypes=[("Resimler", "*.png *.jpg *.jpeg")])
        if path:
            self.logo_path = path
            try:
                img = Image.open(path)
                img = img.resize((120, 60), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.logo_label.config(image=photo)
                self.logo_label.image = photo
                logger.info(f"Logo yüklendi: {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"Logo yükleme hatası: {str(e)}")
                messagebox.showerror("Hata", f"Logo yüklenemedi: {str(e)}")

    def select_folder(self, folder_type):
        """Klasör seçimi yapar."""
        path = filedialog.askdirectory(title=f"{'Şirket Dokümanları' if folder_type == 'doc' else 'Kanun PDF'} Klasörü Seç")
        if path:
            if folder_type == "doc":
                self.doc_folder = path
                self.doc_label.config(text=os.path.basename(path))
            else:
                self.law_folder = path
                self.law_label.config(text=os.path.basename(path))
            logger.info(f"{'Doküman' if folder_type == 'doc' else 'Kanun'} klasörü seçildi: {path}")

    def select_output_file(self):
        """Çıktı dosyası seçimi yapar."""
        formats = {
            "txt": [("Text Files", "*.txt")],
            "json": [("JSON Files", "*.json")],
            "csv": [("CSV Files", "*.csv")],
            "html": [("HTML Files", "*.html")]
        }
        path = filedialog.asksaveasfilename(
            title="Analiz Sonucunu Kaydet",
            defaultextension=f".{self.format_var.get()}",
            filetypes=formats[self.format_var.get()]
        )
        if path:
            self.output_file = path
            self.output_label.config(text=os.path.basename(path))
            logger.info(f"Çıktı dosyası seçildi: {path}")

    def start_analysis(self):
        """Analizi başlatır."""
        if not self.doc_folder or not self.law_folder or not self.output_file:
            logger.warning("Doküman, kanun klasörleri veya çıktı dosyası eksik")
            messagebox.showwarning("Uyarı", "Lütfen doküman, kanun klasörlerini ve çıktı dosyasını seçin")
            return
        self.cancel_flag = False
        threading.Thread(target=self.perform_analysis, daemon=True).start()

    def cancel_analysis(self):
        """Analizi iptal eder."""
        self.cancel_flag = True
        logger.info("Analiz iptal ediliyor...")
        messagebox.showinfo("Bilgi", "Analiz iptal edildi")

    def perform_analysis(self):
        """Analiz sürecini yürütür."""
        try:
            self.progress["value"] = 0
            self.status_label.config(text="Analiz başlatılıyor...")
            logger.info("Analiz süreci başlatıldı")

            if not os.path.exists(self.doc_folder) or not os.path.exists(self.law_folder):
                raise ValueError("Seçilen klasörler mevcut değil")

            # Şirket dokümanlarını yükleme
            company_docs = self.load_documents(self.doc_folder)
            if not company_docs or self.cancel_flag:
                return

            self.progress["value"] = 40
            self.status_label.config(text="Şirket dokümanları yüklendi")

            # Kanun dokümanlarını yükleme
            law_docs = self.load_law_documents(self.law_folder)
            if not law_docs or self.cancel_flag:
                return

            self.progress["value"] = 60
            self.status_label.config(text="Kanun dokümanları yüklendi")

            # Metin analizi
            self.analysis_result = self.analyze_texts(company_docs, law_docs)
            if not self.analysis_result or self.cancel_flag:
                return

            self.progress["value"] = 80
            self.status_label.config(text="Metin analizi tamamlandı")

            # Rapor oluşturma
            self.generate_report()
            self.save_to_database()

            self.progress["value"] = 100
            self.status_label.config(text="Analiz tamamlandı")
            logger.info("Analiz başarıyla tamamlandı")

            if messagebox.askyesno("Analiz Tamamlandı", "Raporu şimdi açmak ister misiniz?"):
                self.open_report()

        except Exception as e:
            logger.error(f"Analiz hatası: {str(e)}")
            messagebox.showerror("Hata", f"Analiz sırasında hata oluştu: {str(e)}")
        finally:
            self.progress["value"] = 0

    def load_documents(self, folder_path):
        """Dokümanları yükler."""
        supported_formats = ['.pdf', '.txt', '.csv', '.json', '.html', '.md']
        documents = []
        total_files = 0
        max_file_size = 100 * 1024 * 1024  # 100 MB sınır

        for root, _, files in os.walk(folder_path):
            for file in files:
                if self.cancel_flag:
                    return None
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_formats:
                    total_files += 1
                    file_path = os.path.join(root, file)
                    if os.path.getsize(file_path) > max_file_size:
                        logger.warning(f"{file} çok büyük, atlanıyor")
                        continue
                    try:
                        content = self.read_file(file_path, ext)
                        documents.append({
                            "path": file_path,
                            "name": file,
                            "type": ext[1:],
                            "content": content,
                            "size": len(content)
                        })
                    except Exception as e:
                        logger.warning(f"{file} işlenirken hata: {str(e)}")
        logger.info(f"{total_files} doküman yüklendi ({len(documents)} başarılı)")
        return documents

    def read_file(self, file_path, extension):
        """Dosyadan metin çıkarır."""
        if extension not in ['.pdf', '.txt', '.csv', '.json', '.html', '.md']:
            raise ValueError("Desteklenmeyen dosya formatı")
        safe_path = os.path.normpath(file_path)
        base_folder = self.doc_folder if extension in ['.txt', '.csv', '.json', '.html', '.md'] else self.law_folder
        if not safe_path.startswith(os.path.normpath(base_folder)):
            raise ValueError("Geçersiz dosya yolu")
        try:
            if extension == '.pdf':
                return self.extract_pdf_text(file_path)
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif extension == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
            elif extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.dumps(json.load(f))
            elif extension == '.html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    return soup.get_text()
            elif extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return markdown.markdown(f.read())
        except Exception as e:
            raise Exception(f"Dosya okuma hatası: {str(e)}")

    def extract_pdf_text(self, file_path):
        """PDF dosyasından metin çıkarır."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except:
            return pdfminer_extract(file_path)

    def load_law_documents(self, folder_path):
        """Kanun dokümanlarını yükler."""
        law_docs = []
        for file in os.listdir(folder_path):
            if self.cancel_flag:
                return None
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(folder_path, file)
                try:
                    content = self.extract_pdf_text(file_path)
                    law_docs.append({
                        "name": file,
                        "content": content,
                        "articles": self.extract_articles(content)
                    })
                except Exception as e:
                    logger.warning(f"{file} işlenirken hata: {str(e)}")
        logger.info(f"{len(law_docs)} kanun dokümanı yüklendi")
        return law_docs

    def extract_articles(self, text):
        """Metinden madde başlıklarını ve içeriklerini çıkarır."""
        pattern = r'(?i)((?:madde|md|m\.|bölüm|kanun|maddesi)\s*[\d\-–:]+)'
        articles = re.split(pattern, text)
        result = []
        for i in range(1, len(articles), 2):
            if i+1 < len(articles):
                result.append({
                    "title": articles[i].strip(),
                    "content": articles[i+1].strip()
                })
        return result

    def analyze_texts(self, company_docs, law_docs):
        """Metinleri analiz eder."""
        combined_text = "\n\n".join([doc['content'] for doc in company_docs])
        try:
            lang = detect(combined_text[:500])
            stop_words = set(stopwords.words(lang)) if lang in stopwords.fileids() else set()
            logger.info(f"Dil algılandı: {lang}, {len(stop_words)} stop word kullanılıyor")
        except:
            stop_words = set(stopwords.words('turkish'))
            logger.warning("Dil algılanamadı, Türkçe stop word'ler kullanılıyor")

        word_freq = defaultdict(int)
        words = nltk.word_tokenize(combined_text.lower())
        for word in words:
            if word.isalpha() and len(word) > 2 and word not in stop_words:
                word_freq[word] += 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        audit_title = self.audit_var.get()
        keywords = self.audit_keywords.get(audit_title, [])
        compliance_results = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for law in law_docs:
                law_name = law['name']
                for article in law['articles']:
                    article['source'] = law_name
                    futures.append(executor.submit(
                        self.analyze_article_compliance,
                        article,
                        combined_text,
                        keywords
                    ))
            for future in as_completed(futures):
                if self.cancel_flag:
                    return None
                compliance_results.append(future.result())

        return {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "doc_folder": self.doc_folder,
            "law_folder": self.law_folder,
            "total_docs": len(company_docs),
            "total_laws": len(law_docs),
            "total_articles": sum(len(law['articles']) for law in law_docs),
            "total_chars": len(combined_text),
            "top_words": top_words,
            "compliance_results": compliance_results,
            "audit_title": audit_title
        }

    def analyze_article_compliance(self, article, company_text, keywords):
        """Madde uyumluluğunu analiz eder."""
        content = article['content'].lower()
        result = {
            "law_name": article.get('source', 'Bilinmeyen Kanun'),
            "article_title": article['title'],
            "compliance": "UYUMSUZ",
            "company_match": False,
            "missing_keywords": ", ".join([kw for kw in keywords if kw not in content]) or "Yok",
            "confidence": 0.0
        }
        if self.bert_active.get() and self.bert_model:
            try:
                inputs = self.bert_tokenizer(content[:512], return_tensors="pt", truncation=True)
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).tolist()[0]
                result['compliance'] = "UYUMLU" if probs[1] > 0.5 else "UYUMSUZ"
                result['confidence'] = probs[1] if probs[1] > 0.5 else probs[0]
            except Exception as e:
                logger.error(f"BERT analizi hatası: {str(e)}")
        else:
            result['company_match'] = any(keyword in company_text.lower() for keyword in keywords if keyword in content)
            result['compliance'] = "UYUMLU" if result['company_match'] else "UYUMSUZ"
        return result

    def generate_report(self):
        """Seçilen formata göre rapor oluşturur."""
        if not any(self.report_options[opt].get() for opt in self.report_options):
            logger.warning("Hiçbir rapor içeriği seçilmedi")
            messagebox.showwarning("Uyarı", "Lütfen en az bir rapor içeriği seçin")
            return
        fmt = self.format_var.get()
        try:
            if fmt == "txt":
                self.generate_txt_report()
            elif fmt == "json":
                self.generate_json_report()
            elif fmt == "csv":
                self.generate_csv_report()
            elif fmt == "html":
                self.generate_html_report()
            logger.info(f"Rapor oluşturuldu: {self.output_file}")
        except Exception as e:
            logger.error(f"Rapor oluşturma hatası: {str(e)}")
            messagebox.showerror("Hata", f"Rapor oluşturulamadı: {str(e)}")

    def generate_txt_report(self):
        """TXT formatında rapor oluşturur."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("HUKUKİ METİN ANALİZ VE DENETİM RAPORU\n")
            f.write("="*70 + "\n\n")
            if self.report_options["Genel Bilgiler"].get():
                f.write(f"Rapor Tarihi: {self.analysis_result['date']}\n")
                f.write(f"Şirket Dokümanları: {self.analysis_result['doc_folder']}\n")
                f.write(f"Kanun Klasörü: {self.analysis_result['law_folder']}\n")
                f.write(f"Toplam Doküman: {self.analysis_result['total_docs']}\n")
                f.write(f"Toplam Kanun: {self.analysis_result['total_laws']}\n")
                f.write(f"Toplam Madde: {self.analysis_result['total_articles']}\n")
                f.write(f"Denetim Başlığı: {self.analysis_result['audit_title']}\n\n")
            if self.report_options["Kelime Frekansları"].get():
                f.write("En Sık Geçen Kelimeler:\n")
                f.write("-"*50 + "\n")
                for word, freq in self.analysis_result['top_words']:
                    f.write(f"{word}: {freq} kez\n")
                f.write("\n")
            if self.report_options["Uyumluluk Durumu"].get():
                f.write("Uyumluluk Analizi:\n")
                f.write("-"*50 + "\n")
                for result in self.analysis_result['compliance_results']:
                    f.write(f"{result['law_name']} - {result['article_title']}: {result['compliance']}\n")
                    if self.report_options["Eksiklik Tespiti"].get() and result['missing_keywords'] != "Yok":
                        f.write(f"  Eksik Anahtar Kelimeler: {result['missing_keywords']}\n")
                    if self.report_options["Karşılaştırmalı Analiz"].get():
                        f.write(f"  Şirkette Bulunma: {'Evet' if result['company_match'] else 'Hayır'}\n")
                    if self.report_options["Örnek Madde"].get():
                        f.write(f"  Güven Skoru: {result['confidence']:.2f}\n")
                    f.write("\n")

    def generate_json_report(self):
        """JSON formatında rapor oluşturur."""
        report_data = {
            "report_date": self.analysis_result['date'],
            "doc_folder": self.analysis_result['doc_folder'],
            "law_folder": self.analysis_result['law_folder'],
            "total_docs": self.analysis_result['total_docs'],
            "total_laws": self.analysis_result['total_laws'],
            "total_articles": self.analysis_result['total_articles'],
            "audit_title": self.analysis_result['audit_title'],
            "top_words": [{"word": word, "frequency": freq} for word, freq in self.analysis_result['top_words']] if self.report_options["Kelime Frekansları"].get() else [],
            "compliance_results": [
                {k: v for k, v in result.items() if k != "confidence" or self.report_options["Örnek Madde"].get()}
                for result in self.analysis_result['compliance_results']
            ] if self.report_options["Uyumluluk Durumu"].get() else []
        }
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON raporu oluşturuldu: {self.output_file}")

    def generate_csv_report(self):
        """CSV formatında rapor oluşturur."""
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if self.report_options["Genel Bilgiler"].get():
                writer.writerow(["Rapor Tarihi", self.analysis_result['date']])
                writer.writerow(["Şirket Dokümanları", self.analysis_result['doc_folder']])
                writer.writerow(["Kanun Klasörü", self.analysis_result['law_folder']])
                writer.writerow(["Toplam Doküman", self.analysis_result['total_docs']])
                writer.writerow(["Toplam Kanun", self.analysis_result['total_laws']])
                writer.writerow(["Toplam Madde", self.analysis_result['total_articles']])
                writer.writerow(["Denetim Başlığı", self.analysis_result['audit_title']])
                writer.writerow([])
            if self.report_options["Kelime Frekansları"].get():
                writer.writerow(["Kelime", "Frekans"])
                for word, freq in self.analysis_result['top_words']:
                    writer.writerow([word, freq])
                writer.writerow([])
            if self.report_options["Uyumluluk Durumu"].get():
                writer.writerow(["Kanun", "Madde", "Uyum", "Şirkette Bulunma", "Eksik Kelimeler", "Güven Skoru"])
                for result in self.analysis_result['compliance_results']:
                    writer.writerow([
                        result['law_name'], result['article_title'], result['compliance'],
                        'Evet' if result['company_match'] else 'Hayır' if self.report_options["Karşılaştırmalı Analiz"].get() else '',
                        result['missing_keywords'] if self.report_options["Eksiklik Tespiti"].get() else '',
                        f"{result['confidence']:.2f}" if self.report_options["Örnek Madde"].get() else ''
                    ])
        logger.info(f"CSV raporu oluşturuldu: {self.output_file}")

    def generate_html_report(self):
        """HTML formatında modern rapor oluşturur."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset='UTF-8'>
    <title>Hukuki Metin Analiz Raporu</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 20px; padding: 15px; border-left: 4px solid #0d6efd; background-color: white; }
        .compliance-table { width: 100%; margin-top: 10px; }
        .compliant { color: green; font-weight: bold; }
        .non-compliant { color: red; font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
""")
            if self.logo_path and self.report_options["Genel Bilgiler"].get():
                try:
                    with open(self.logo_path, "rb") as img_file:
                        logo_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    f.write(f'<img src="data:image/png;base64,{logo_base64}" class="img-fluid" style="max-height:80px;"><br>\n')
                except:
                    pass
            f.write("""
        <h1>Hukuki Metin Analiz ve Denetim Raporu</h1>
    </div>
""")
            if self.report_options["Genel Bilgiler"].get():
                f.write("""
    <div class="section">
        <h2>Genel Bilgiler</h2>
        <p><strong>Rapor Tarihi:</strong> {}</p>
        <p><strong>Şirket Dokümanları:</strong> {}</p>
        <p><strong>Kanun Klasörü:</strong> {}</p>
        <p><strong>Toplam Doküman:</strong> {}</p>
        <p><strong>Toplam Kanun:</strong> {}</p>
        <p><strong>Toplam Madde:</strong> {}</p>
        <p><strong>Denetim Başlığı:</strong> {}</p>
    </div>
""".format(
                    self.analysis_result['date'], self.analysis_result['doc_folder'],
                    self.analysis_result['law_folder'], self.analysis_result['total_docs'],
                    self.analysis_result['total_laws'], self.analysis_result['total_articles'],
                    self.analysis_result['audit_title']
                ))
            if self.report_options["Kelime Frekansları"].get():
                f.write("""
    <div class="section">
        <h2>Kelime Frekans Analizi</h2>
        <table class="table table-striped compliance-table">
            <thead><tr><th>Kelime</th><th>Frekans</th></tr></thead>
            <tbody>
""")
                for word, freq in self.analysis_result['top_words']:
                    f.write(f"<tr><td>{word}</td><td>{freq}</td></tr>\n")
                f.write("""
            </tbody>
        </table>
    </div>
""")
            if self.report_options["Uyumluluk Durumu"].get():
                f.write("""
    <div class="section">
        <h2>Uyumluluk Analizi</h2>
        <table class="table table-striped compliance-table">
            <thead><tr><th>Kanun</th><th>Madde</th><th>Uyum</th><th>Şirkette Bulunma</th><th>Eksik Kelimeler</th><th>Güven Skoru</th></tr></thead>
            <tbody>
""")
                for result in self.analysis_result['compliance_results']:
                    compliance_class = "compliant" if result['compliance'] == "UYUMLU" else "non-compliant"
                    f.write(f"""
                <tr>
                    <td>{result['law_name']}</td>
                    <td>{result['article_title']}</td>
                    <td class="{compliance_class}">{result['compliance']}</td>
                    <td>{'Evet' if result['company_match'] else 'Hayır' if self.report_options["Karşılaştırmalı Analiz"].get() else ''}</td>
                    <td>{result['missing_keywords'] if self.report_options["Eksiklik Tespiti"].get() else ''}</td>
                    <td>{f"{result['confidence']:.2f}" if self.report_options["Örnek Madde"].get() else ''}</td>
                </tr>
""")
                f.write("""
            </tbody>
        </table>
    </div>
""")
            f.write("""
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
""")
        logger.info(f"HTML raporu oluşturuldu: {self.output_file}")

    def save_to_database(self):
        """Analiz sonuçlarını veritabanına kaydeder."""
        if self.cursor is None or self.conn is None:
            logger.warning("Veritabanı bağlantısı yok, kayıt yapılmadı")
            return
        try:
            with self.db_lock:
                self.cursor.execute('''
                    INSERT INTO analysis_results (
                        analysis_date, doc_folder, law_folder, total_docs, total_laws,
                        total_articles, total_chars, result_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.analysis_result['date'],
                    self.analysis_result['doc_folder'],
                    self.analysis_result['law_folder'],
                    self.analysis_result['total_docs'],
                    self.analysis_result['total_laws'],
                    self.analysis_result['total_articles'],
                    self.analysis_result['total_chars'],
                    self.output_file
                ))
                self.conn.commit()
            self.load_history()
            logger.info("Analiz sonuçları veritabanına kaydedildi")
        except (sqlite3.Error, PermissionError) as e:
            logger.error(f"Veritabanı kayıt hatası: {str(e)}")
            messagebox.showerror("Hata", f"Veritabanına kaydedilemedi: {str(e)}")

    def load_history(self):
        """Geçmiş analizleri yükler."""
        if self.cursor is None or self.conn is None:
            logger.warning("Veritabanı bağlantısı yok, geçmiş yüklenemedi")
            return
        try:
            self.history_tree.delete(*self.history_tree.get_children())
            self.cursor.execute("SELECT id, analysis_date, doc_folder, law_folder, result_file FROM analysis_results")
            for row in self.cursor.fetchall():
                self.history_tree.insert("", tk.END, values=row)
        except (sqlite3.Error, PermissionError) as e:
            logger.error(f"Geçmiş yükleme hatası: {str(e)}")
            messagebox.showerror("Hata", f"Geçmiş yüklenemedi: {str(e)}")

    def open_report(self):
        """Rapor dosyasını açar."""
        if not self.output_file or not os.path.exists(self.output_file):
            logger.warning("Rapor dosyası bulunamadı")
            messagebox.showwarning("Uyarı", "Rapor dosyası bulunamadı")
            return
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(self.output_file)
            elif system == "Darwin":
                subprocess.call(('open', self.output_file))
            else:
                subprocess.call(('xdg-open', self.output_file))
        except:
            try:
                webbrowser.open(self.output_file)
            except Exception as e:
                logger.error(f"Rapor açma hatası: {str(e)}")
                messagebox.showerror("Hata", f"Rapor açılamadı: {str(e)}")

    def open_history_report(self, event):
        """Geçmiş analiz raporunu açar."""
        selected = self.history_tree.selection()
        if selected:
            item = self.history_tree.item(selected[0])
            file_path = item['values'][4]
            if os.path.exists(file_path):
                self.output_file = file_path
                self.open_report()
            else:
                logger.warning("Rapor dosyası bulunamadı")
                messagebox.showwarning("Uyarı", "Rapor dosyası bulunamadı")

    def show_graph(self):
        """Kelime frekansları ve uyumluluk oranları için grafik gösterir."""
        if not self.analysis_result:
            logger.warning("Analiz sonuçları yok")
            messagebox.showwarning("Uyarı", "Önce bir analiz yapmalısınız")
            return
        try:
            words, freqs = zip(*self.analysis_result['top_words'])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Kelime frekansları
            bars = ax1.bar(words, freqs, color='#0d6efd')
            ax1.set_title('En Sık Geçen Kelimeler')
            ax1.set_xlabel('Kelimeler')
            ax1.set_ylabel('Frekans')
            ax1.tick_params(axis='x', rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')
            
            # Uyumluluk oranları
            compliant = sum(1 for r in self.analysis_result['compliance_results'] if r['compliance'] == "UYUMLU")
            non_compliant = len(self.analysis_result['compliance_results']) - compliant
            ax2.pie([compliant, non_compliant], labels=['Uyumlu', 'Uyumsuz'], colors=['#28a745', '#dc3545'], autopct='%1.1f%%')
            ax2.set_title('Uyumluluk Oranları')
            
            plt.tight_layout()
            
            graph_window = tk.Toplevel(self.root)
            graph_window.title("Analiz Grafikleri")
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            toolbar = NavigationToolbar2Tk(canvas, graph_window)
            toolbar.update()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            graph_window.grid_rowconfigure(0, weight=1)
            graph_window.grid_columnconfigure(0, weight=1)
            
            logger.info("Analiz grafikleri gösterildi")
        except Exception as e:
            logger.error(f"Grafik oluşturma hatası: {str(e)}")
            messagebox.showerror("Hata", f"Grafik oluşturulamadı: {str(e)}")

    def log(self, message, level="info"):
        """Log mesajını kaydeder ve gösterir."""
        logger.log(getattr(logging, level.upper()), message)
        if hasattr(self, 'log_text'):
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] [{level.upper()}] {message}\n")
            if level == "error":
                self.log_text.tag_add("error", "end-2c linestart", "end-1c lineend")
                self.log_text.tag_config("error", foreground="red")
            elif level == "warning":
                self.log_text.tag_add("warning", "end-2c linestart", "end-1c lineend")
                self.log_text.tag_config("warning", foreground="orange")
            self.log_text.config(state=tk.DISABLED)
            self.log_text.yview(tk.END)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EnhancedLegalAnalyzer(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Uygulama başlatma hatası: {str(e)}")
        messagebox.showerror("Hata", f"Uygulama başlatılamadı: {str(e)}")