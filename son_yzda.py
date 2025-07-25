import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import re
import spacy
from spacy.lang.tr.stop_words import STOP_WORDS
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline
import pandas as pd
from fpdf import FPDF
import logging

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('yzda_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('YZDA')

# 1. KANUN BİLGİSİ VE ANALİZ MOTORU
class LawKnowledgeEngine:
    def __init__(self, law_dir):
        self.law_dir = law_dir
        self.laws = {}
        self.knowledge_graph = nx.DiGraph()
        
        # Spacy Türkçe model kontrolü
        try:
            self.nlp = spacy.load("tr_core_news_lg")
            logger.info("tr_core_news_lg modeli başarıyla yüklendi")
        except:
            try:
                self.nlp = spacy.load("tr_core_news_sm")
                logger.info("tr_core_news_sm modeli başarıyla yüklendi")
            except:
                logger.warning("Temel Türkçe model kullanılıyor, bazı özellikler sınırlı olacak")
                self.nlp = spacy.blank("tr")
        
        self.load_laws()
        self.build_knowledge_graph()
        logger.info("Kanun bilgi motoru başlatıldı")
    
    def load_laws(self):
        """Kanun dosyalarını yükle ve işle (alt klasörleri de tarar)"""
        if not os.path.exists(self.law_dir):
            logger.error(f"Kanun klasörü bulunamadı: {self.law_dir}")
            return
            
        for root, dirs, files in os.walk(self.law_dir):
            for filename in files:
                if filename.lower().endswith(('.pdf', '.txt', '.docx')):
                    filepath = os.path.join(root, filename)
                    law_name = os.path.splitext(filename)[0]
                    
                    try:
                        if filename.lower().endswith('.pdf'):
                            text = self.extract_pdf_text(filepath)
                        else:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read()
                        
                        if not text.strip():
                            logger.warning(f"{filename} dosyası boş görünüyor")
                            continue
                        
                        self.laws[law_name] = self.parse_law_structure(text, law_name)
                        logger.info(f"{law_name} yüklendi ({len(text)} karakter)")
                    except Exception as e:
                        logger.error(f"{filename} yüklenirken hata: {str(e)}")
    
    def add_new_law(self, filepath):
        """Yeni kanun dosyası ekle"""
        try:
            filename = os.path.basename(filepath)
            law_name = os.path.splitext(filename)[0]
            if filename.lower().endswith('.pdf'):
                text = self.extract_pdf_text(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if not text.strip():
                logger.warning(f"{filename} dosyası boş görünüyor")
                return False
            
            self.laws[law_name] = self.parse_law_structure(text, law_name)
            self.build_knowledge_graph()  # Grafiği güncelle
            logger.info(f"Yeni kanun eklendi: {law_name}")
            return True
        except Exception as e:
            logger.error(f"Yeni kanun eklenirken hata: {str(e)}")
            return False
    
    def extract_pdf_text(self, filepath):
        """PDF'den metin çıkar"""
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            logger.error(f"PDF okuma hatası: {str(e)}")
        return text
    
    def parse_law_structure(self, text, law_name):
        """Kanun metnini yapılandır"""
        structure = {
            "name": law_name,
            "articles": {},
            "keywords": []
        }
        
        article_pattern = r'(?i)((?:madde|md|m\.|maddesi)\s*\d+[a-z]*)\s*(.*?)(?=(?:(?:madde|md|m\.|maddesi)\s*\d+|\Z))'
        articles = re.findall(article_pattern, text, re.DOTALL)
        
        for title, content in articles:
            article_id = re.search(r'\d+[a-z]*', title).group()
            structure["articles"][article_id] = {
                "title": title.strip(),
                "content": content.strip(),
                "requirements": self.extract_requirements(content)
            }
        
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = set(STOP_WORDS)
        keywords = [word for word in words if word not in stop_words]
        structure["keywords"] = list(set(keywords))[:100]
        
        return structure
    
    def extract_requirements(self, text):
        """Madde içindeki gereklilikleri çıkar"""
        requirements = []
        requirement_pattern = r'(?i)(?:gerekir|zorunludur|yükümlüdür|mecburidir)'
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if re.search(requirement_pattern, sentence):
                requirements.append(sentence.strip())
        
        return requirements
    
    def build_knowledge_graph(self):
        """Kanunlar arası ilişki grafiği oluştur"""
        self.knowledge_graph.clear()
        for law in self.laws:
            self.knowledge_graph.add_node(law, type="law", **self.laws[law])
        
        for law_name, law_data in self.laws.items():
            for article_id, article in law_data["articles"].items():
                node_id = f"{law_name}_{article_id}"
                self.knowledge_graph.add_node(node_id, type="article", **article)
                self.knowledge_graph.add_edge(law_name, node_id)
                
                for other_law in self.laws:
                    if law_name != other_law:
                        for keyword in self.laws[other_law]["keywords"]:
                            if keyword in article["content"].lower():
                                self.knowledge_graph.add_edge(node_id, other_law)
                                self.knowledge_graph.add_edge(other_law, node_id)
                                break
        
        logger.info(f"İlişki grafiği oluşturuldu: {len(self.knowledge_graph.nodes)} düğüm, {len(self.knowledge_graph.edges)} kenar")
    
    def get_law_names(self):
        """Mevcut kanun isimlerini döndür"""
        return list(self.laws.keys())
    
    def get_law_articles(self, law_name):
        """Bir kanunun maddelerini döndür"""
        if law_name in self.laws:
            return list(self.laws[law_name]["articles"].keys())
        return []

# 2. ÇAPRAZ ANALİZ VE ÖNERİ MOTORU
class CrossLawAnalyzer:
    def __init__(self, knowledge_engine):
        self.knowledge_engine = knowledge_engine
        self.nlp = knowledge_engine.nlp
        
        try:
            self.recommendation_engine = pipeline("text-generation", model="dbmdz/gpt2-turkish-cased")
        except:
            try:
                self.recommendation_engine = pipeline("text-generation", model="redrussianarmy/gpt2-turkish-cased")
            except:
                logger.warning("Öneri motoru kullanılamıyor, basit öneriler üretilecek")
                self.recommendation_engine = None
                
        logger.info("Çapraz analiz motoru başlatıldı")
    
    def analyze_document(self, doc_text, company_name, selected_laws):
        """Belgeyi seçili kanunlar açısından analiz et"""
        results = {
            "company": company_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compliance": {},
            "cross_issues": [],
            "recommendations": []
        }
        
        for law_name in selected_laws:
            if law_name in self.knowledge_engine.laws:
                results["compliance"][law_name] = {}
                law_data = self.knowledge_engine.laws[law_name]
                
                for article_id, article in law_data["articles"].items():
                    compliance_score = self.calculate_compliance(
                        doc_text, 
                        article["content"],
                        article["requirements"]
                    )
                    
                    cross_issues = self.find_cross_issues(
                        f"{law_name}_{article_id}", 
                        doc_text
                    )
                    
                    results["compliance"][law_name][article_id] = {
                        "score": compliance_score,
                        "status": "UYUMLU" if compliance_score >= 0.8 else "KISMEN UYUMLU" if compliance_score >= 0.5 else "UYUMSUZ",
                        "cross_issues": cross_issues
                    }
                    
                    if compliance_score < 0.8 or cross_issues:
                        results["recommendations"].extend(
                            self.generate_recommendations(law_name, article_id, compliance_score, cross_issues)
                        )
        
        results["recommendations"] = sorted(results["recommendations"], 
                                           key=lambda x: x["priority"], 
                                           reverse=True)[:10]
        
        logger.info(f"{company_name} için analiz tamamlandı")
        return results
    
    def calculate_compliance(self, doc_text, article_text, requirements):
        """Uyum skorunu hesapla"""
        try:
            doc_words = set(word.lower() for word in re.findall(r'\b\w+\b', doc_text))
            article_words = set(word.lower() for word in re.findall(r'\b\w+\b', article_text))
            common_words = doc_words & article_words
            keyword_similarity = len(common_words) / max(1, len(article_words))
            
            requirement_coverage = sum(
                1 for req in requirements 
                if any(keyword in doc_text.lower() for keyword in re.findall(r'\b\w+\b', req.lower()))
            ) / max(1, len(requirements))
            
            return (keyword_similarity + requirement_coverage) / 2
        except Exception as e:
            logger.error(f"Uyum skoru hesaplanırken hata: {str(e)}")
            return 0.0
    
    def find_cross_issues(self, article_node, doc_text):
        """Çapraz mevzuat sorunlarını bul"""
        cross_issues = []
        
        try:
            if article_node in self.knowledge_engine.knowledge_graph:
                for neighbor in self.knowledge_engine.knowledge_graph.neighbors(article_node):
                    if "type" in self.knowledge_engine.knowledge_graph.nodes[neighbor]:
                        neighbor_type = self.knowledge_engine.knowledge_graph.nodes[neighbor]["type"]
                        if neighbor_type == "article":
                            content = self.knowledge_engine.knowledge_graph.nodes[neighbor]["content"]
                            keywords = re.findall(r'\b\w{4,}\b', content.lower())
                            
                            found_keywords = sum(1 for kw in keywords if kw in doc_text.lower())
                            coverage = found_keywords / max(1, len(keywords))
                            
                            if coverage < 0.4:
                                cross_issues.append({
                                    "related_article": neighbor,
                                    "description": self.knowledge_engine.knowledge_graph.nodes[neighbor]["title"]
                                })
        except Exception as e:
            logger.error(f"Çapraz sorunlar bulunurken hata: {str(e)}")
        
        return cross_issues
    
    def generate_recommendations(self, law_name, article_id, compliance_score, cross_issues):
        """Akıllı öneriler oluştur"""
        recommendations = []
        priority = max(1, int(10 * (1 - compliance_score)))
        
        recommendations.append({
            "priority": priority,
            "law": law_name,
            "article": article_id,
            "type": "temel",
            "description": self.generate_ai_recommendation(law_name, article_id, compliance_score)
        })
        
        for issue in cross_issues:
            recommendations.append({
                "priority": min(10, priority + 2),
                "law": "Çapraz Mevzuat",
                "article": f"{law_name} {article_id} ↔ {issue['related_article']}",
                "type": "çapraz",
                "description": f"{issue['related_article']} maddesi ile ilişkili düzenlemeler gözden geçirilmeli"
        })
        
        return recommendations
    
    def generate_ai_recommendation(self, law_name, article_id, compliance_score):
        """Yapay zeka veya basit kural tabanlı öneri oluştur"""
        if not self.recommendation_engine:
            if compliance_score < 0.5:
                return f"{law_name} kanununun {article_id} maddesi için ciddi uyumsuzluk tespit edildi. İlgili maddeyi gözden geçirin ve gerekli düzenlemeleri yapın. (Uzman denetçi tarafından kontrol önerilir)"
            elif compliance_score < 0.8:
                return f"{law_name} kanununun {article_id} maddesi için kısmi uyumsuzluk tespit edildi. İyileştirme yapılması önerilir. (Uzman denetçi tarafından kontrol önerilir)"
            else:
                return f"{law_name} kanununun {article_id} maddesi uyumlu görünmektedir. Yine de periyodik kontroller önerilir."
        
        prompt = f"{law_name} kanununun {article_id} numaralı maddesinde "
        if compliance_score < 0.5:
            prompt += "ciddi uyumsuzluk tespit edildi. Düzeltici aksiyon önerileri:"
        elif compliance_score < 0.8:
            prompt += "kısmi uyumsuzluk tespit edildi. İyileştirme önerileri:"
        else:
            prompt += "uyum sağlanmış olmakla birlikte iyileştirme önerileri:"
        
        try:
            ai_response = self.recommendation_engine(
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']
            return ai_response.split(":")[-1].strip() + " (Uzman denetçi tarafından kontrol önerilir)"
        except Exception as e:
            logger.error(f"Yapay zeka önerisi oluşturulamadı: {str(e)}")
            return "Yapay zeka öneri oluşturamadı. Lütfen manuel kontrol edin."

# 3. RAPORLAMA MODÜLÜ
class AuditReporter:
    def __init__(self):
        self.logger = logging.getLogger('YZDA.Reporter')
        self.logger.info("Raporlama modülü başlatıldı")
    
    def generate_report(self, analysis_result, output_path, format="pdf"):
        """Analiz sonuçlarını seçilen formatta raporla"""
        try:
            if format == "pdf":
                return self.generate_pdf_report(analysis_result, output_path)
            elif format == "html":
                return self.generate_html_report(analysis_result, output_path)
            elif format == "csv":
                return self.generate_csv_report(analysis_result, output_path)
            elif format == "json":
                return self.generate_json_report(analysis_result, output_path)
            elif format == "txt":
                return self.generate_txt_report(analysis_result, output_path)
            else:
                raise ValueError("Geçersiz rapor formatı")
        except Exception as e:
            self.logger.error(f"Rapor oluşturma hatası: {str(e)}")
            return False
    
    def generate_pdf_report(self, analysis_result, output_path):
        """PDF rapor oluştur"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="YAPAY ZEKA DENETİM ASİSTANI (YZDA) RAPORU", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Şirket: {analysis_result['company']}", ln=True)
        pdf.cell(200, 10, txt=f"Tarih: {analysis_result['timestamp']}", ln=True)
        pdf.cell(200, 10, txt="Not: Bu rapor otomatik oluşturulmuştur ve uzman denetçi tarafından kontrol edilmelidir.", ln=True)
        pdf.ln(10)
        
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Uyumluluk Özeti", ln=True)
        pdf.set_font("Arial", size=12)
        
        for law, articles in analysis_result["compliance"].items():
            compliant_count = sum(1 for a in articles.values() if a["status"] == "UYUMLU")
            total_count = len(articles)
            pdf.cell(200, 10, txt=f"{law}: {compliant_count}/{total_count} madde uyumlu", ln=True)
        
        pdf.ln(10)
        
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Detaylı Uyum Analizi", ln=True)
        pdf.set_font("Arial", size=10)
        
        for law, articles in analysis_result["compliance"].items():
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, txt=f"{law} Kanunu", ln=True)
            pdf.set_font("Arial", size=10)
            
            for article_id, data in articles.items():
                status = data["status"]
                pdf.cell(50, 8, txt=f"Madde {article_id}", border=1)
                pdf.cell(40, 8, txt=status, border=1)
                pdf.cell(30, 8, txt=f"Skor: {data['score']:.2f}", border=1)
                
                cross_issues = ", ".join([issue['related_article'] for issue in data["cross_issues"]]) if data["cross_issues"] else "Yok"
                pdf.cell(70, 8, txt=f"Çapraz Sorunlar: {cross_issues}", border=1, ln=True)
        
        pdf.ln(10)
        
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="YZDA Önerileri", ln=True)
        pdf.set_font("Arial", size=10)
        
        for i, rec in enumerate(analysis_result["recommendations"], 1):
            pdf.set_font("Arial", "B", 10)
            pdf.cell(200, 8, txt=f"{i}. [{rec['priority']}/10] {rec['law']} - {rec['article']}", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 6, txt=rec["description"])
            pdf.ln(2)
        
        pdf.output(output_path)
        self.logger.info(f"PDF rapor oluşturuldu: {output_path}")
        return True
    
    def generate_html_report(self, analysis_result, output_path):
        """HTML rapor oluştur"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>YZDA Raporu - {analysis_result['company']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .non-compliant {{ background-color: #ffdddd; }}
                .partial {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <h1>Yapay Zeka Denetim Asistanı (YZDA) Raporu</h1>
            <h2>Şirket: {analysis_result['company']}</h2>
            <p>Rapor Tarihi: {analysis_result['timestamp']}</p>
            <p><strong>Not:</strong> Bu rapor otomatik oluşturulmuştur ve uzman denetçi tarafından kontrol edilmelidir.</p>
            
            <h2>Uyumluluk Özeti</h2>
            <ul>
        """
        
        for law, articles in analysis_result["compliance"].items():
            compliant_count = sum(1 for a in articles.values() if a["status"] == "UYUMLU")
            total_count = len(articles)
            html_content += f"<li><strong>{law}:</strong> {compliant_count}/{total_count} madde uyumlu</li>"
        
        html_content += """
            </ul>
            
            <h2>Detaylı Uyum Analizi</h2>
            <table>
                <tr>
                    <th>Kanun</th>
                    <th>Madde</th>
                    <th>Durum</th>
                    <th>Skor</th>
                    <th>Çapraz Sorunlar</th>
                </tr>
        """
        
        for law, articles in analysis_result["compliance"].items():
            for article_id, data in articles.items():
                status_class = ""
                if data["status"] == "UYUMSUZ":
                    status_class = "non-compliant"
                elif data["status"] == "KISMEN UYUMLU":
                    status_class = "partial"
                
                cross_issues = ", ".join([issue['related_article'] for issue in data["cross_issues"]]) if data["cross_issues"] else "Yok"
                
                html_content += f"""
                <tr class="{status_class}">
                    <td>{law}</td>
                    <td>{article_id}</td>
                    <td>{data['status']}</td>
                    <td>{data['score']:.2f}</td>
                    <td>{cross_issues}</td>
                </tr>
                """
        
        html_content += """
            </table>
            
            <h2>YZDA Önerileri</h2>
            <ol>
        """
        
        for rec in analysis_result["recommendations"]:
            html_content += f"""
            <li>
                <p><strong>[Öncelik: {rec['priority']}/10] {rec['law']} - {rec['article']}</strong></p>
                <p>{rec['description']}</p>
            </li>
            """
        
        html_content += """
            </ol>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML rapor oluşturuldu: {output_path}")
        return True
    
    def generate_csv_report(self, analysis_result, output_path):
        """CSV rapor oluştur"""
        rows = []
        rows.append(["Kanun", "Madde", "Durum", "Skor", "Çapraz Sorunlar"])
        
        for law, articles in analysis_result["compliance"].items():
            for article_id, data in articles.items():
                cross_issues = "; ".join([issue['related_article'] for issue in data["cross_issues"]]) if data["cross_issues"] else ""
                rows.append([
                    law,
                    article_id,
                    data["status"],
                    f"{data['score']:.2f}",
                    cross_issues
                ])
        
        rows.append([])
        rows.append(["ÖNERİLER"])
        rows.append(["Öncelik", "Kanun", "Madde", "Öneri"])
        
        for rec in analysis_result["recommendations"]:
            rows.append([
                rec["priority"],
                rec["law"],
                rec["article"],
                rec["description"]
            ])
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, header=False)
        self.logger.info(f"CSV rapor oluşturuldu: {output_path}")
        return True
    
    def generate_json_report(self, analysis_result, output_path):
        """JSON rapor oluştur"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON rapor oluşturuldu: {output_path}")
        return True
    
    def generate_txt_report(self, analysis_result, output_path):
        """TXT rapor oluştur"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"YAPAY ZEKA DENETİM ASİSTANI (YZDA) RAPORU\n")
            f.write(f"Şirket: {analysis_result['company']}\n")
            f.write(f"Tarih: {analysis_result['timestamp']}\n")
            f.write(f"Not: Bu rapor otomatik oluşturulmuştur ve uzman denetçi tarafından kontrol edilmelidir.\n\n")
            
            f.write("UYUMLULUK ÖZETİ\n")
            f.write("="*50 + "\n")
            for law, articles in analysis_result["compliance"].items():
                compliant_count = sum(1 for a in articles.values() if a["status"] == "UYUMLU")
                total_count = len(articles)
                f.write(f"{law}: {compliant_count}/{total_count} madde uyumlu\n")
            
            f.write("\nDETAYLI ANALİZ\n")
            f.write("="*50 + "\n")
            for law, articles in analysis_result["compliance"].items():
                f.write(f"\n{law} KANUNU\n")
                f.write("-"*50 + "\n")
                for article_id, data in articles.items():
                    cross_issues = ", ".join([issue['related_article'] for issue in data["cross_issues"]]) if data["cross_issues"] else "Yok"
                    f.write(f"Madde {article_id}: {data['status']} (Skor: {data['score']:.2f})\n")
                    f.write(f"  Çapraz Sorunlar: {cross_issues}\n")
            
            f.write("\nYZDA ÖNERİLERİ\n")
            f.write("="*50 + "\n")
            for rec in analysis_result["recommendations"]:
                f.write(f"\n[{rec['priority']}/10] {rec['law']} - {rec['article']}\n")
                f.write(f"{rec['description']}\n")
        
        self.logger.info(f"TXT rapor oluşturuldu: {output_path}")
        return True

# 4. ARAYÜZ VE UYGULAMA KATMANI
class YZDA_GUI(tk.Tk):
    def __init__(self, law_dir):
        super().__init__()
        self.title("Yapay Zeka Denetim Asistanı (YZDA)")
        self.geometry("1000x700")
        self.law_dir = law_dir
        self.knowledge_engine = None
        self.analyzer = None
        self.reporter = AuditReporter()
        self.current_company = None
        self.selected_laws = []
        self.analysis_result = None
        
        self.add_logo()
        self.create_widgets()
        self.load_law_knowledge()
        logger.info("YZDA arayüzü başlatıldı")
    
    def add_logo(self):
        """Logo ekle"""
        logo_path = r"C:\Users\Sami\Desktop\YZDA\logo.png"
        try:
            if os.path.exists(logo_path):
                self.logo_image = Image.open(logo_path)
                self.logo_image = self.logo_image.resize((300, 100), Image.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(self.logo_image)
                self.logo_label = ttk.Label(self, image=self.logo_photo)
                self.logo_label.pack(side=tk.TOP, pady=10)
                logger.info("Logo başarıyla yüklendi")
            else:
                logger.warning("Logo dosyası bulunamadı")
        except Exception as e:
            logger.error(f"Logo yüklenirken hata: {str(e)}")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Denetim Kontrolleri")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Firma Seçimi:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.company_var = tk.StringVar()
        self.company_combo = ttk.Combobox(control_frame, textvariable=self.company_var, state="readonly", width=25)
        self.company_combo.grid(row=0, column=1, padx=5, pady=5)
        self.company_combo.bind("<<ComboboxSelected>>", self.on_company_select)
        ttk.Button(control_frame, text="Firma Ekle", command=self.add_company).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Kanun Seçimi:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.law_var = tk.StringVar()
        self.law_combo = ttk.Combobox(control_frame, textvariable=self.law_var, width=50)  # Genişlik iki katına çıkarıldı
        self.law_combo.grid(row=1, column=1, padx=5, pady=5)
        self.law_combo["state"] = "disabled"
        
        self.add_law_btn = ttk.Button(control_frame, text="Ekle", command=self.add_law)
        self.add_law_btn.grid(row=1, column=2, padx=5, pady=5)
        self.add_law_btn["state"] = "disabled"
        
        ttk.Button(control_frame, text="Yeni Kanun Yükle", command=self.upload_new_law).grid(row=2, column=2, padx=5, pady=5)
        
        self.selected_laws_listbox = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=5, width=35)
        self.selected_laws_listbox.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W+tk.E)
        ttk.Button(control_frame, text="Seçimi Kaldır", command=self.remove_law).grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Rapor Formatı:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.report_format = tk.StringVar(value="pdf")
        formats = [("PDF", "pdf"), ("HTML", "html"), ("CSV", "csv"), ("JSON", "json"), ("Metin", "txt")]
        for i, (text, value) in enumerate(formats):
            ttk.Radiobutton(control_frame, text=text, variable=self.report_format, value=value).grid(
                row=5+i//2, column=1+i%2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(control_frame, text="Rapor Kayıt Yeri:").grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
        self.report_path = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.report_path, state="readonly", width=25).grid(
            row=8, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        ttk.Button(control_frame, text="Gözat", command=self.browse_report_path).grid(row=8, column=2, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Denetimi Başlat", command=self.start_audit, 
                  style="Accent.TButton").grid(row=9, column=0, columnspan=3, padx=5, pady=10)
        
        notebook = ttk.Notebook(display_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Log Akışı")
        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        rec_frame = ttk.Frame(notebook)
        notebook.add(rec_frame, text="YZDA Önerileri")
        self.rec_text = scrolledtext.ScrolledText(rec_frame, state="disabled")
        self.rec_text.pack(fill=tk.BOTH, expand=True)
        
        preview_frame = ttk.Frame(notebook)
        notebook.add(preview_frame, text="Rapor Önizleme")
        self.preview_text = scrolledtext.ScrolledText(preview_frame, state="normal")
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=('Arial', 10, 'bold'), foreground='white', background='#2c3e50')  # Koyu arka plan
    
    def load_law_knowledge(self):
        """Kanun bilgisini yükle"""
        self.log("Kanun bilgisi yükleniyor...")
        try:
            self.knowledge_engine = LawKnowledgeEngine(self.law_dir)
            self.analyzer = CrossLawAnalyzer(self.knowledge_engine)
            
            law_names = self.knowledge_engine.get_law_names()
            if law_names:
                self.law_combo["values"] = law_names
                self.law_combo["state"] = "readonly"
                self.add_law_btn["state"] = "normal"
                self.log(f"{len(law_names)} kanun başarıyla yüklendi")
            else:
                self.log("Yüklenebilir kanun bulunamadı", "warning")
                self.law_combo["state"] = "disabled"
            
        except Exception as e:
            self.log(f"Kanun bilgisi yüklenirken hata: {str(e)}", "error")
            self.law_combo["state"] = "disabled"
    
    def add_company(self):
        """Firma ekle"""
        file_path = filedialog.askopenfilename(
            title="Firma Dokümanı Seç",
            filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            company_name = os.path.splitext(os.path.basename(file_path))[0]
            current_values = list(self.company_combo["values"])
            if company_name not in current_values:
                current_values.append(company_name)
                self.company_combo["values"] = current_values
            self.company_var.set(company_name)
            self.current_company = {"name": company_name, "path": file_path}
            self.log(f"{company_name} firması eklendi")
    
    def on_company_select(self, event):
        """Firma seçildiğinde"""
        company_name = self.company_var.get()
        self.current_company = {"name": company_name}
        self.log(f"{company_name} firması seçildi")
    
    def add_law(self):
        """Kanun ekle"""
        law_name = self.law_var.get()
        if law_name and law_name not in self.selected_laws:
            self.selected_laws.append(law_name)
            self.selected_laws_listbox.insert(tk.END, law_name)
            self.log(f"{law_name} kanunu eklendi")
    
    def upload_new_law(self):
        """Yeni kanun dosyası yükle"""
        file_path = filedialog.askopenfilename(
            title="Yeni Kanun Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            if self.knowledge_engine.add_new_law(file_path):
                law_names = self.knowledge_engine.get_law_names()
                self.law_combo["values"] = law_names
                self.law_combo["state"] = "readonly"
                self.add_law_btn["state"] = "normal"
                self.log(f"Yeni kanun başarıyla yüklendi: {os.path.basename(file_path)}")
            else:
                self.log("Yeni kanun yüklenemedi", "error")
    
    def remove_law(self):
        """Seçili kanunu kaldır"""
        selected_indices = self.selected_laws_listbox.curselection()
        for i in selected_indices[::-1]:
            law_name = self.selected_laws_listbox.get(i)
            self.selected_laws_listbox.delete(i)
            self.selected_laws.remove(law_name)
            self.log(f"{law_name} kanunu kaldırıldı")
    
    def browse_report_path(self):
        """Rapor kayıt yeri seç"""
        formats = {
            "pdf": [("PDF Files", "*.pdf")],
            "html": [("HTML Files", "*.html")],
            "csv": [("CSV Files", "*.csv")],
            "json": [("JSON Files", "*.json")],
            "txt": [("Text Files", "*.txt")]
        }
        fmt = self.report_format.get()
        file_path = filedialog.asksaveasfilename(
            title="Raporu Kaydet",
            defaultextension=f".{fmt}",
            filetypes=formats.get(fmt, [("All Files", "*.*")])
        )
        if file_path:
            self.report_path.set(file_path)
    
    def start_audit(self):
        """Denetimi başlat"""
        if not self.current_company:
            self.log("Lütfen bir firma seçin", "warning")
            return
        if not self.selected_laws:
            self.log("Lütfen en az bir kanun seçin", "warning")
            return
        if not self.report_path.get():
            self.log("Lütfen rapor kayıt yerini seçin", "warning")
            return
        
        self.log("Denetim başlatılıyor...")
        threading.Thread(target=self.run_audit, daemon=True).start()
    
    def run_audit(self):
        """Denetimi gerçekleştir (arka planda)"""
        try:
            if not self.current_company.get("path"):
                self.log("Firma doküman yolu bulunamadı", "error")
                return
            
            if self.current_company["path"].endswith('.pdf'):
                with fitz.open(self.current_company["path"]) as doc:
                    doc_text = ""
                    for page in doc:
                        doc_text += page.get_text()
            else:
                with open(self.current_company["path"], 'r', encoding='utf-8') as f:
                    doc_text = f.read()
            
            if len(doc_text) < 100:
                self.log("Doküman içeriği çok kısa, analiz yapılamıyor", "warning")
                return
            
            self.log(f"Doküman yüklendi ({len(doc_text)} karakter), analiz başlıyor...")
            
            analysis_result = self.analyzer.analyze_document(
                doc_text, 
                self.current_company["name"], 
                self.selected_laws
            )
            self.analysis_result = analysis_result
            
            self.show_recommendations(analysis_result)
            
            self.log("Rapor oluşturuluyor...")
            if self.reporter.generate_report(
                analysis_result, 
                self.report_path.get(), 
                self.report_format.get()
            ):
                self.log("Rapor başarıyla oluşturuldu")
                self.preview_report()
                # Rapor tamamlandı mesajı ve görüntüleme seçeneği
                response = messagebox.askyesno(
                    "Rapor Tamamlandı",
                    f"Rapor başarıyla oluşturuldu: {self.report_path.get()}\nRaporu şimdi görüntülemek ister misiniz?"
                )
                if response:
                    os.startfile(self.report_path.get())  # Windows-specific
            else:
                self.log("Rapor oluşturulamadı", "error")
            
        except Exception as e:
            self.log(f"Denetim sırasında hata: {str(e)}", "error")
    
    def show_recommendations(self, analysis_result):
        """Önerileri görüntüle"""
        self.rec_text.config(state=tk.NORMAL)
        self.rec_text.delete(1.0, tk.END)
        
        self.rec_text.insert(tk.END, "YZDA ÖNERİLERİ\n", "title")
        self.rec_text.insert(tk.END, "="*50 + "\n\n")
        
        for rec in analysis_result["recommendations"]:
            self.rec_text.insert(tk.END, f"[Öncelik: {rec['priority']}/10] ", "priority")
            self.rec_text.insert(tk.END, f"{rec['law']} - {rec['article']}\n", "subtitle")
            self.rec_text.insert(tk.END, f"{rec['description']}\n\n")
        
        self.rec_text.tag_configure("title", font=("Arial", 14, "bold"))
        self.rec_text.tag_configure("priority", font=("Arial", 10, "bold"), foreground="red")
        self.rec_text.tag_configure("subtitle", font=("Arial", 10, "bold"))
        self.rec_text.config(state=tk.DISABLED)
    
    def preview_report(self):
        """Rapor önizlemesini göster"""
        if not self.analysis_result:
            return
        
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        
        self.preview_text.insert(tk.END, "RAPOR ÖNİZLEME\n", "title")
        self.preview_text.insert(tk.END, "="*50 + "\n\n")
        
        self.preview_text.insert(tk.END, f"Şirket: {self.analysis_result['company']}\n")
        self.preview_text.insert(tk.END, f"Tarih: {self.analysis_result['timestamp']}\n")
        self.preview_text.insert(tk.END, f"Not: Bu rapor otomatik oluşturulmuştur ve uzman denetçi tarafından kontrol edilmelidir.\n\n")
        
        self.preview_text.insert(tk.END, "UYUMLULUK ÖZETİ\n", "header")
        self.preview_text.insert(tk.END, "="*50 + "\n")
        for law, articles in self.analysis_result["compliance"].items():
            compliant_count = sum(1 for a in articles.values() if a["status"] == "UYUMLU")
            total_count = len(articles)
            self.preview_text.insert(tk.END, f"{law}: {compliant_count}/{total_count} madde uyumlu\n")
        
        self.preview_text.insert(tk.END, "\nÖNERİLER\n", "header")
        self.preview_text.insert(tk.END, "="*50 + "\n")
        for rec in self.analysis_result["recommendations"]:
            self.preview_text.insert(tk.END, f"\n[{rec['priority']}/10] {rec['law']} - {rec['article']}\n")
            self.preview_text.insert(tk.END, f"{rec['description']}\n")
        
        self.preview_text.tag_configure("title", font=("Arial", 16, "bold"))
        self.preview_text.tag_configure("header", font=("Arial", 12, "bold"))
        self.preview_text.config(state=tk.DISABLED)
    
    def log(self, message, level="info"):
        """Log mesajını arayüze ve dosyaya yaz"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        tag = "info"
        if level == "error":
            logger.error(message)
            tag = "error"
        elif level == "warning":
            logger.warning(message)
            tag = "warning"
        else:
            logger.info(message)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_msg + "\n", tag)
        
        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("warning", foreground="orange")
        self.log_text.tag_config("error", foreground="red")
        
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

# 5. UYGULAMAYI BAŞLAT
if __name__ == "__main__":
    LAW_DIR = r"C:\Users\Sami\Desktop\YZDA\kanunlar"
    app = YZDA_GUI(LAW_DIR)
    app.mainloop()