#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║              3D INTERACTIVE EMBEDDING VISUALIZER v2.1                      ║
║         50D Embedding'leri 3D Uzaya Dönüştür ve İnteraktif Göster          ║
╚════════════════════════════════════════════════════════════════════════════╝

YENİ ÖZELLİKLER:
════════════════
• X, Y, Z eksenleri belirgin şekilde gösterilir
• İsteğe bağlı vektör çizimi (--draw-vectors)
• Her noktadan orijine vektör çizgileri
• Vektörler yarı saydam (alpha=0.3) gösterilir
• Eksen etiketleri daha belirgin

KULLANIM:
═════════
# Normal kullanım (vektörsüz)
python visualize_embeddings_3d.py --db embeddings.db

# Vektörleri de göster
python visualize_embeddings_3d.py --db embeddings.db --draw-vectors

# PCA ile vektörler
python visualize_embeddings_3d.py --db embeddings.db --method pca --draw-vectors

# t-SNE ile vektörler (orijinal uzayda mantıklı değil ama gösterir)
python visualize_embeddings_3d.py --db embeddings.db --method tsne --draw-vectors
"""

import sqlite3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.widgets import TextBox
import sys

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# =========================
# VERİTABANI OKUMA
# =========================
def load_embeddings_from_db(db_name):
    """Veritabanından embedding'leri yükle"""
    try:
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        
        cur.execute("SELECT word, vector FROM embeddings")
        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            print(f"[ERROR] Veritabanında embedding yok: {db_name}")
            return None, None
        
        words = []
        vectors = []
        
        for word, vector_str in rows:
            try:
                vec = [float(x) for x in vector_str.split(",")]
                words.append(word)
                vectors.append(vec)
            except ValueError:
                print(f"[WARN] Geçersiz vector formatı: {word}")
                continue
        
        print(f"[INFO] {len(words)} embedding yüklendi (boyut: {len(vectors[0])}D)")
        return np.array(vectors), np.array(words)
    
    except Exception as e:
        print(f"[ERROR] Veritabanı okuma hatası: {e}")
        return None, None

# =========================
# BOYUT İNDİRGEME YÖNTEMLERİ
# =========================
def reduce_to_3d_pca(embeddings):
    """PCA kullanarak 50D → 3D dönüştürme"""
    print("[INFO] PCA ile 3D'ye dönüştürülüyor...")
    pca = PCA(n_components=3)
    result = pca.fit_transform(embeddings)
    
    total_var = np.sum(pca.explained_variance_ratio_)
    print(f"[INFO] Açıklanan varyans: {total_var*100:.2f}%")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    
    return result

def reduce_to_3d_tsne(embeddings):
    """t-SNE kullanarak 50D → 3D dönüştürme (yavaş ama iyi)"""
    print("[INFO] t-SNE ile 3D'ye dönüştürülüyor (bu biraz zaman alabilir)...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1), max_iter=1000)
    result = tsne.fit_transform(embeddings)
    return result

def reduce_to_3d_umap(embeddings):
    """UMAP kullanarak 50D → 3D dönüştürme (hızlı ve iyi)"""
    if not HAS_UMAP:
        print("[ERROR] UMAP yüklü değil. Kurmak için: pip install umap-learn")
        return None
    
    print("[INFO] UMAP ile 3D'ye dönüştürülüyor...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    result = reducer.fit_transform(embeddings)
    return result

def reduce_to_3d_pca_with_variance(embeddings, variance_threshold=0.9):
    """
    PCA + Varimax döndürme
    İlk 3 PC'yi seç ama maksimum varyansı kapsasın
    """
    print(f"[INFO] PCA + Variance (hedef: {variance_threshold*100:.0f}%) ile 3D'ye dönüştürülüyor...")
    
    pca_full = PCA()
    pca_full.fit(embeddings)
    
    # Varyansın belirtilen % oranını kaplayan bileşen sayısını bul
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    n_components = min(n_components, 3)
    
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(embeddings)
    
    if n_components < 3:
        # Kalan boyutları rastgele ekle (sadece görselleştirme için)
        result = np.hstack([result, np.random.randn(result.shape[0], 3 - n_components) * 0.01])
    
    print(f"[INFO] Kullanılan bileşen sayısı: {n_components}")
    for i in range(min(n_components, 3)):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%")
    
    return result

# =========================
# KOSİN BENZERLİĞİ
# =========================
def cosine_similarity(v1, v2):
    """İki vektör arasında kosinüs benzerliği hesapla"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)

def find_similar_words(word, words, original_embeddings, top_n=5):
    """Verilen kelimeye benzer kelimeleri bul"""
    if word not in words:
        return []
    
    word_idx = np.where(words == word)[0][0]
    word_vec = original_embeddings[word_idx]
    
    similarities = []
    for i, w in enumerate(words):
        if w == word:
            continue
        
        sim = cosine_similarity(word_vec, original_embeddings[i])
        similarities.append((w, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# =========================
# İNTERAKTİF VİZÜALİZASYON
# =========================
class Interactive3DVisualizer:
    def __init__(self, embeddings_3d, words, original_embeddings, draw_vectors=False):
        self.embeddings_3d = embeddings_3d
        self.words = words
        self.original_embeddings = original_embeddings
        self.selected_word = None
        self.selected_indices = []
        self.draw_vectors = draw_vectors
        
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle("3D Interactive Embedding Visualizer", fontsize=16, fontweight='bold')
        
        # 3D grafik
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.scatter = None
        self.vector_lines = []
        self.text_annotations = []
        
        # İstatistik paneli
        self.ax_stats = self.fig.add_subplot(122)
        self.ax_stats.axis('off')
        
        # Arama kutusu
        ax_search = plt.axes([0.15, 0.05, 0.3, 0.04])
        self.text_box = TextBox(ax_search, 'Kelime ara:', initial='')
        self.text_box.on_submit(self.on_search)
        
        self.draw_3d()
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
    
    def draw_3d(self):
        """3D grafik çiz"""
        self.ax_3d.clear()
        
        # Eksenleri belirginleştir
        self.ax_3d.set_xlabel('X EKSENİ', fontsize=12, fontweight='bold', labelpad=10)
        self.ax_3d.set_ylabel('Y EKSENİ', fontsize=12, fontweight='bold', labelpad=10)
        self.ax_3d.set_zlabel('Z EKSENİ', fontsize=12, fontweight='bold', labelpad=10)
        
        # Eksen çizgilerini kalınlaştır
        self.ax_3d.xaxis.line.set_linewidth(2)
        self.ax_3d.yaxis.line.set_linewidth(2)
        self.ax_3d.zaxis.line.set_linewidth(2)
        
        # Eksen renkleri
        self.ax_3d.xaxis.label.set_color('red')
        self.ax_3d.yaxis.label.set_color('green')
        self.ax_3d.zaxis.label.set_color('blue')
        
        # Grid ekle
        self.ax_3d.grid(True, alpha=0.3)
        
        colors = ['red' if idx in self.selected_indices else 'blue' for idx in range(len(self.words))]
        sizes = [100 if idx in self.selected_indices else 30 for idx in range(len(self.words))]
        
        self.scatter = self.ax_3d.scatter(
            self.embeddings_3d[:, 0],
            self.embeddings_3d[:, 1],
            self.embeddings_3d[:, 2],
            c=colors,
            s=sizes,
            alpha=0.7,
            picker=True,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Vektör çizimleri (opsiyonel)
        if self.draw_vectors:
            print("[INFO] Vektörler çiziliyor...")
            for i, (x, y, z) in enumerate(self.embeddings_3d):
                # Orijinden (0,0,0) noktaya vektör çiz
                color = 'red' if i in self.selected_indices else 'gray'
                line = self.ax_3d.plot([0, x], [0, y], [0, z], 
                                      color=color, alpha=0.3, linewidth=1)
                self.vector_lines.append(line)
        
        # Eksen merkezine bir nokta koy (orijin)
        self.ax_3d.scatter([0], [0], [0], c='black', s=50, marker='o', alpha=0.5)
        
        self.ax_3d.set_title("3D Embedding Uzayı\n(Tıkla = Seç, Ara = Benzerliği Göster)", 
                           fontsize=12, fontweight='bold')
        
        # Seçilen kelimeleri etiketle
        for idx in self.selected_indices:
            self.ax_3d.text(
                self.embeddings_3d[idx, 0],
                self.embeddings_3d[idx, 1],
                self.embeddings_3d[idx, 2],
                self.words[idx],
                fontsize=9,
                fontweight='bold',
                color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
    
    def on_pick(self, event):
        """Noktaya tıklama olayı"""
        if event.artist != self.scatter:
            return
        
        indices = event.ind
        
        if len(indices) == 0:
            return
        
        idx = indices[0]
        word = self.words[idx]
        
        # Toggle seçim
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.append(idx)
        
        self.selected_word = word
        self.update_display()
    
    def on_search(self, text):
        """Arama kutusundan kelime ara"""
        if not text.strip():
            self.selected_indices = []
            self.update_display()
            return
        
        matches = np.where([text.lower() in w.lower() for w in self.words])[0]
        
        if len(matches) == 0:
            print(f"[INFO] '{text}' kelimesi bulunamadı")
            return
        
        self.selected_indices = list(matches)
        self.selected_word = text
        self.update_display()
    
    def update_display(self):
        """Grafik ve istatistikleri güncelle"""
        self.vector_lines = []  # Eski vektörleri temizle
        self.draw_3d()
        self.draw_stats()
        plt.draw()
    
    def draw_stats(self):
        """İstatistik panelini çiz"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        text_content = "📊 BENZERLİK ANALİZİ\n" + "="*40 + "\n\n"
        
        # Vektör çizim durumu
        text_content += f"Vektör Çizimi: {'AÇIK' if self.draw_vectors else 'KAPALI'}\n"
        text_content += "-"*40 + "\n\n"
        
        if len(self.selected_indices) == 0:
            text_content += "Grafikteki noktaya tıkla\nveya kelime adını ara"
        else:
            for idx in self.selected_indices[:3]:  # İlk 3'ü göster
                word = self.words[idx]
                text_content += f"🔍 '{word}'\n"
                text_content += "-" * 40 + "\n"
                
                # Benzer kelimeleri bul
                similar = find_similar_words(word, self.words, self.original_embeddings, top_n=5)
                
                for sim_word, sim_score in similar:
                    text_content += f"  • {sim_word:20s} {sim_score:6.3f}\n"
                
                # 50D vektör istatistikleri
                vec = self.original_embeddings[idx]
                text_content += f"\n📈 Vektör İstatistikleri:\n"
                text_content += f"  Norm: {np.linalg.norm(vec):.4f}\n"
                text_content += f"  Min: {np.min(vec):.4f}\n"
                text_content += f"  Max: {np.max(vec):.4f}\n"
                text_content += f"  Mean: {np.mean(vec):.4f}\n"
                text_content += f"  Std: {np.std(vec):.4f}\n\n"
        
        self.ax_stats.text(
            0.05, 0.95,
            text_content,
            transform=self.ax_stats.transAxes,
            fontfamily='monospace',
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def show(self):
        """Göster"""
        self.draw_stats()
        plt.tight_layout()
        plt.show()

# =========================
# ANA PROGRAM
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Interactive Embedding Visualizer - 50D vektörleri 3D uzayda gösterir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔═ YENİ ÖZELLİKLER ══════════════════════════════════════════════════════════╗

VEKTÖR ÇİZİMİ (--draw-vectors):
────────────────────────────────
Bu parametre ile her noktadan orijine (0,0,0) bir vektör çizilir.
Vektörler yarı saydam (alpha=0.3) gösterilir.

Anlamı:
• Vektörün yönü = Kelimenin anlam yönü
• Vektörün uzunluğu = Kelimenin "şiddeti" (norm)
• Seçili kelimelerin vektörleri kırmızı, diğerleri gri

EKSENLER:
────────
• X EKSENİ (Kırmızı) - Birinci ana bileşen
• Y EKSENİ (Yeşil)   - İkinci ana bileşen  
• Z EKSENİ (Mavi)    - Üçüncü ana bileşen

NOT: t-SNE ve UMAP'te vektörler orijinal uzayı temsil etmez,
sadece görsel amaçlıdır. PCA'da anlamlıdır.
        """
    )
    
    parser.add_argument(
        '-d', '--db',
        type=str,
        default='embeddings.db',
        help="Embedding veritabanı dosyası (default: embeddings.db)"
    )
    
    parser.add_argument(
        '-m', '--method',
        type=str,
        default='pca',
        choices=['pca', 'tsne', 'umap', 'pca-variance'],
        help="Boyut indirme yöntemi (default: pca)"
    )
    
    parser.add_argument(
        '-v', '--variance',
        type=float,
        default=0.95,
        help="PCA-variance için hedef varyans oranı (default: 0.95)"
    )
    
    parser.add_argument(
        '--draw-vectors',
        action='store_true',
        help="""Vektörleri çiz (opsiyonel)
        
        Bu parametre eklendiğinde her noktadan orijine (0,0,0) bir vektör çizilir.
        Vektörler yarı saydam çizgilerle gösterilir.
        
        Örnek: --draw-vectors
        """
    )
    
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════╗
║    3D INTERACTIVE EMBEDDING VISUALIZER     ║
║         50D → 3D Dönüştürme                ║
╚════════════════════════════════════════════╝
""")
    
    # Vektör çizim durumunu göster
    if args.draw_vectors:
        print("[INFO] Vektör çizimi AKTİF - Her noktaya vektör çizilecek")
    else:
        print("[INFO] Vektör çizimi PASIF (--draw-vectors ile aktifleştir)")
    
    # Veritabanından oku
    print(f"[INFO] Veritabanı okunuyor: {args.db}")
    embeddings_50d, words = load_embeddings_from_db(args.db)
    
    if embeddings_50d is None:
        sys.exit(1)
    
    print(f"[INFO] Toplam: {len(words)} kelime, {embeddings_50d.shape[1]}D vektörler\n")
    
    # 3D'ye dönüştür
    if args.method == 'pca':
        embeddings_3d = reduce_to_3d_pca(embeddings_50d)
    elif args.method == 'tsne':
        embeddings_3d = reduce_to_3d_tsne(embeddings_50d)
    elif args.method == 'umap':
        embeddings_3d = reduce_to_3d_umap(embeddings_50d)
        if embeddings_3d is None:
            print("[ERROR] UMAP başarısız, PCA kullanılıyor...")
            embeddings_3d = reduce_to_3d_pca(embeddings_50d)
    elif args.method == 'pca-variance':
        embeddings_3d = reduce_to_3d_pca_with_variance(embeddings_50d, args.variance)
    
    print(f"[INFO] 3D vektörleri hazır (boyut: {embeddings_3d.shape})\n")
    
    # İnteraktif visualizer başlat
    print("[INFO] İnteraktif görselleştirici açılıyor...")
    print("  → Noktaya tıkla: Seç / Deseç")
    print("  → Arama kutusuna kelime yaz: Benzer kelimeleri göster")
    print("  → Mouse ile döndür: 3D görüntüyü hareket ettir")
    print("  → Eksenler: X(Kırmızı), Y(Yeşil), Z(Mavi)\n")
    
    viz = Interactive3DVisualizer(embeddings_3d, words, embeddings_50d, args.draw_vectors)
    viz.show()