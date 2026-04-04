import pandas as pd
from openpyxl import load_workbook
import win32com.client as win32
import os
from tqdm import tqdm
import sys

# === Dosya yolları ===
kandela_path = 'Kandela_2025.xlsx'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# === Veriyi yükle ===
data = pd.read_excel(kandela_path)

# === Excel COM objesini bir kez başlat (performans için) ===
excel = win32.Dispatch("Excel.Application")
excel.Visible = False

try:
    # Satırları ilerleme çubuğuyla işle
    for index, row in tqdm(data.iterrows(), total=len(data), desc="PDF Fatura Oluşturuluyor", file=sys.stdout):

        # --- Temel kontroller ---
        if pd.isna(row.get('DATE')):
            print(f"Atlanıyor (satır {index + 1}): DATE eksik.")
            continue
        if pd.isna(row.get('Name')):
            print(f"Atlanıyor (satır {index + 1}): Name eksik.")
            continue

        # --- Döviz türüne göre şablonu seç ---
        if pd.notna(row.get('TRY')):
            invoice_template_path = 'invoice.xlsx'
            currency_value = row['TRY']
        elif pd.notna(row.get('Pound')):
            invoice_template_path = 'Invoice_Pound.xlsx'
            currency_value = row['Pound']
        elif pd.notna(row.get('Euro')):
            invoice_template_path = 'Invoice_Euro.xlsx'
            currency_value = row['Euro']
        else:
            print(f"Atlanıyor (satır {index + 1}): Döviz değeri bulunamadı.")
            continue

        # --- Şablonu yükle ---
        workbook = load_workbook(invoice_template_path)
        sheet = workbook.active

        # --- Hücrelere veri aktar ---
        sheet['D4'].value = row['DATE']           # Tarih
        sheet['D7'].value = row['Inv No']         # Fatura numarası
        sheet['A9'].value = row['Name']           # İsim
        sheet['C15'].value = row.get('Hours', '') # Saat bilgisi
        sheet['D15'].value = currency_value       # Döviz değeri

        # ✅ A15 hücresine Kandela'daki I sütunundaki bilgi yaz
        sheet['A15'].value = row.iloc[8]  # Excel'de I sütunu 9. sütun (index 8)

        # --- Alt klasör ve dosya adını hazırla ---
        date_str = pd.to_datetime(row['DATE']).strftime('%d.%m')
        month_folder = pd.to_datetime(row['DATE']).strftime('%m')
        subfolder_path = os.path.join(output_folder, month_folder)
        os.makedirs(subfolder_path, exist_ok=True)

        # Güvenli dosya adı oluştur
        safe_name = "".join(c for c in str(row['Name']) if c.isalnum() or c in " _-").strip()
        filename = f"{safe_name} - {date_str}.pdf"
        output_path = os.path.join(subfolder_path, filename)

        # --- Geçici Excel dosyasını kaydet ---
        temp_invoice_path = 'temporary_invoice.xlsx'
        workbook.save(temp_invoice_path)
        workbook.close()

        # --- PDF'e dönüştür ---
        try:
            wb = excel.Workbooks.Open(os.path.abspath(temp_invoice_path))
            wb.ExportAsFixedFormat(0, os.path.abspath(output_path))  # 0 = PDF
            wb.Close(False)
        except Exception as e:
            print(f"Hata (satır {index + 1}, {safe_name}): {e}")
        finally:
            if os.path.exists(temp_invoice_path):
                os.remove(temp_invoice_path)

finally:
    # Excel her durumda kapatılır
    excel.Quit()

print("✅ PDF üretimi tamamlandı. Dosyalar 'output' klasöründe, aya göre sınıflandırıldı.")
