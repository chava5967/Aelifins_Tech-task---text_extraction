python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

to run 

python app.py --input sample_invoice.jpg --output ./ocr_out



This project provides a "Python script to extract tables from scanned images (and PDFs when converted to images)" using OpenCV and Tesseract OCR.  
It detects table cells, organizes them into rows and columns, and exports extracted data into "JSON, CSV, and HTML" formats, along with a debug visualization that shows detected table cells.  


## Features  

- **Image preprocessing** (grayscale, resize, contrast, sharpness enhancement).  
- **Table structure detection** using projection profiles + OpenCV morphology.  
- **OCR (Optical Character Recognition)** with [pytesseract](https://pypi.org/project/pytesseract/).  
- **Exports results** in multiple formats:  
  - JSON (raw table data)  
  - CSV (structured rows/columns)  
  - HTML (table rendering)  
  - Debug Image (shows detected cells with bounding boxes).  


## Requirements  

Install dependencies with:  

```bash
pip install pillow numpy opencv-python pytesseract pandas
```

You also need:  
⦁	"Python 3.7+"  
- "Tesseract OCR" installed and configured in your PATH.  
  - On Ubuntu/Debian:  
    ```bash
    sudo apt-get install tesseract-ocr
    ```
  - On Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)  
    and ensure it’s added to your environment variables.  


## Usage  

Run the script with:  

```bash
python ocr_table_extractor.py --input path/to/input_image.png --output ocr_out
```

### Arguments:  

- `--input` : Path to input image (supports PNG, JPG; for PDFs use `pdf2image` to convert to images first).  
- `--output`: Path to output directory (will be created if it does not exist).  


## Output Files  

For each input image, the script generates the following inside the output directory:  

- `<filename>.json` → Extracted table as JSON.  
- `<filename>.csv` → Extracted table as CSV.  
- `<filename>.html` → Extracted table as HTML.  
- `<filename>_debug.png` → Debug image showing detected cell boundaries.  


## Example  

-----bash-----
python ocr_table_extractor.py --input invoice.png --output ./ocr_out


Generates:  
ocr_out/
│── invoice.json
│── invoice.csv
│── invoice.html
│── invoice_debug.png
