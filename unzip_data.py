import zipfile
import os

def unzip_data():
    zip_path = 'archive.zip'
    extract_to = 'hurricane_data'
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return

    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: The zip file is corrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    unzip_data()
