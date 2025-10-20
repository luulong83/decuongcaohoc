# download_vncorenlp.py
import py_vncorenlp

print("Downloading VnCoreNLP models...")
py_vncorenlp.download_model(save_dir='./vncorenlp')
print("Download complete!")