import pandas as pd
from openpyxl import load_workbook
import os
from tqdm import tqdm
import sys
import subprocess # Komut satırı araçlarını çalıştırmak için

# === Dosya yolları ===
kandela_path = 'Kandela_2025.xlsx'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# === PDF Dönüştürme Fonksiyonu (win32com yerine) ===
def convert_xlsx_to_pdf_linux(input_path, output_dir):
    """
    LibreOffice'i kullanarak XLSX dosyasını PDF'e dönüştürür.
    input_path: Geçici olarak kaydedilen XLSX dosyasının tam yolu.
    output_dir: PDF dosyasının kaydedileceği klasör yolu.
    """
    try:
        # LibreOffice'i headless modda (arayüzsüz) çalıştır
        command = [
            'libreoffice',
            '--headless',
            '--convert-to',
            'pdf',
            '--outdir',
            output_dir,
            input_path
        ]
        
        # Komutu çalıştır ve sonucunu bekle
        result = subprocess.run(
            command,
            check=True, # Hata olursa istisna fırlatır
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # LibreOffice, dosyayı input_path'in adını kullanarak output_dir içine kaydeder.
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ PDF dönüştürme hatası: LibreOffice komutu başarısız oldu.")
        print(f"Hata çıktısı:\n{e.stderr.decode('utf-8')}")
        return False
    except FileNotFoundError:
        print("\n❌ Hata: 'libreoffice' komutu bulunamadı. Lütfen LibreOffice'in kurulu olduğundan emin olun.")
        return False

# -------------------------------------------------------------
# === Veriyi yükle ===
try:
    data = pd.read_excel(kandela_path)
except FileNotFoundError:
    print(f"Hata: {kandela_path} dosyası bulunamadı.")
    sys.exit(1)


# === Ana döngü için geçici dosya yolu ===
temp_invoice_path = os.path.join(output_folder, 'temporary_invoice.xlsx')

# --- Ana Döngü ---
# Satırları ilerleme çubuğuyla işle
for index, row in tqdm(data.iterrows(), total=len(data), desc="PDF Fatura Oluşturuluyor", file=sys.stdout):

    # --- Temel kontroller ---
    if pd.isna(row.get('DATE')):
        print(f"\nAtlanıyor (satır {index + 1}): DATE eksik.")
        continue
    if pd.isna(row.get('Name')):
        print(f"\nAtlanıyor (satır {index + 1}): Name eksik.")
        continue

    # --- Döviz türüne göre şablonu seç ---
    if pd.notna(row.get('TRY')):
        invoice_template_path = 'invoice.xlsx'
        currency_value = row['TRY']
    elif pd.notna(row.get('Pound')):
        invoice_template_path = 'invoice_Pound.xlsx'
        currency_value = row['Pound']
    elif pd.notna(row.get('Euro')):
        invoice_template_path = 'invoice_Euro.xlsx'
        currency_value = row['Euro']
    else:
        print(f"\nAtlanıyor (satır {index + 1}): Döviz değeri bulunamadı.")
        continue

    # --- Şablonu yükle ---
    try:
        workbook = load_workbook(invoice_template_path)
    except FileNotFoundError:
        print(f"\n❌ Hata: Şablon dosyası '{invoice_template_path}' bulunamadı.")
        continue
        
    sheet = workbook.active

    # --- Hücrelere veri aktar ---
    sheet['D4'].value = row['DATE']           # Tarih
    sheet['D7'].value = row['Inv No']         # Fatura numarası
    sheet['A9'].value = row['Name']           # İsim
    sheet['C15'].value = row.get('Hours', '') # Saat bilgisi
    sheet['D15'].value = currency_value       # Döviz değeri
    sheet['A15'].value = row.iloc[8]          # Excel'de I sütunu 9. sütun (index 8)

    # --- Alt klasör ve dosya adını hazırla ---
    # PDF adı için gerekli bileşenler
    date_obj = pd.to_datetime(row['DATE'])
    month_folder = date_obj.strftime('%m')
    subfolder_path = os.path.join(output_folder, month_folder)
    os.makedirs(subfolder_path, exist_ok=True)
    
    date_str = date_obj.strftime('%d.%m')
    safe_name = "".join(c for c in str(row['Name']) if c.isalnum() or c in " _-").strip()
    final_pdf_filename = f"{safe_name} - {date_str}.pdf"
    
    
    # --- Geçici Excel dosyasını kaydet ---
    try:
        # openpyxl ile hücrelere yazılan veriyi geçici bir dosyaya kaydet
        workbook.save(temp_invoice_path)
    except Exception as e:
        print(f"\n❌ Geçici dosya kaydetme hatası: {e}")
        continue

    # --- PDF'e dönüştür ve taşı ---
    try:
        # LibreOffice/unoconv kullanarak XLSX'i PDF'e dönüştür
        if convert_xlsx_to_pdf_linux(temp_invoice_path, subfolder_path):
            # Dönüşüm başarılı olursa, LibreOffice tarafından oluşturulan dosyayı bul
            temp_pdf_path = os.path.join(subfolder_path, 'temporary_invoice.pdf')
            final_pdf_path = os.path.join(subfolder_path, final_pdf_filename)
            
            # LibreOffice'in oluşturduğu dosyayı, istediğimiz güvenli isme taşı
            if os.path.exists(temp_pdf_path):
                os.rename(temp_pdf_path, final_pdf_path)
                
    except Exception as e:
        print(f"\n❌ Hata (satır {index + 1}, {safe_name}): Dönüştürme veya taşıma sırasında hata: {e}")
    finally:
        # Geçici Excel dosyasını temizle (her döngüde)
        if os.path.exists(temp_invoice_path):
            os.remove(temp_invoice_path)
            

print("\n✅ PDF üretimi tamamlandı. Dosyalar 'output' klasöründe, aya göre sınıflandırıldı.")