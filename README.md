# Smart Document Processing

This project provides a FastAPI-based document processing service that extracts key invoice-style fields from uploaded images and spreadsheet files.

## Features

- OCR-based text extraction for image documents
- Structured field extraction for `name`, `date`, and `amount`
- CSV and Excel parsing for tabular inputs
- JSON API response for downstream integrations

## Setup Instructions

### Python version

Use **Python 3.11 or newer** for this project.

Check your installed version:

```bash
python --version
```

Expected result:

```text
Python 3.11+
```

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd smart-doc-processing
```

### 2. Create a virtual environment

Use the same supported Python version to create the virtual environment.

#### Windows (cmd)

```cmd
python -m venv venv
venv\Scripts\activate
```

If multiple Python versions are installed, you can explicitly create the environment with Python 3.11:

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Or explicitly with Python 3.11:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run evaluation

To calculate document-level and field-level accuracy against [`data/labels.csv`](data/labels.csv), run:

```bash
python evaluate.py
```

This creates [`outputs/evaluation_summary.json`](outputs/evaluation_summary.json) with:
- exact-match success rate
- field-wise accuracy for `name`, `date`, and `amount`
- failure cases with expected vs predicted values

## How to Run the API

Run the FastAPI app with Uvicorn:

```bash
python api/app.py
```

The API starts on:

```text
http://0.0.0.0:8001
```

Main endpoint:

```text
POST /process-document
```

## Sample Request

Using `curl`:

```bash
curl -X POST "http://127.0.0.1:8001/process-document" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/doc_1.jpg"
```

## Sample Response

```json
{
  "extracted_text": "Invoice\nName: Ravi Patel\nDate: 2024-01-15\nTotal: 1240.50",
  "fields": {
    "name": "Ravi Patel",
    "date": "2024-01-15",
    "amount": "1240.50"
  }
}
```

For spreadsheet inputs, the `fields` values may be arrays when multiple rows contain extractable values.

## Approach Explanation

### 1. OCR pipeline

The OCR logic is implemented in `src/ocr.py`.

- Loads image bytes into an OpenCV image
- Applies preprocessing such as denoising, contrast enhancement, thresholding, cropping, and rotation handling
- Runs PaddleOCR across several image variants and orientations
- Scores OCR candidates and keeps the best normalized text output

### 2. Field extraction

The extraction logic is implemented in `src/extract.py`.

- Cleans noisy OCR text using regex normalization
- Extracts target fields using keyword-based matching and heuristics
- Normalizes dates into `YYYY-MM-DD`
- Normalizes amounts into two-decimal string values
- Supports row-wise extraction for CSV and Excel documents

### 3. API layer

The API is implemented in `api/app.py`.

- Accepts uploaded image, CSV, or Excel files
- Routes image files through OCR + extraction
- Routes spreadsheet files directly through row parsing + extraction
- Returns a consistent JSON structure for clients

### 4. Evaluation

- [`evaluate.py`](evaluate.py:1) loads [`data/labels.csv`](data/labels.csv) and evaluates predictions against the reference labels
- Stores aggregate accuracy and failure details in [`outputs/evaluation_summary.json`](outputs/evaluation_summary.json)
- Keeps request-level prediction examples in [`outputs/sample_results.json`](outputs/sample_results.json)

## Assumptions Made

- Documents are mostly invoice-like and contain recognizable `name`, `date`, and `amount` fields.
- Input images are reasonably readable and contain English text.
- Spreadsheet files contain column names or row values that can be mapped to common aliases such as `name`, `date`, `amount`, `total`, or similar.
- The project is intended for local execution with CPU-based OCR unless GPU support is explicitly enabled.

## Limitations

- Extraction is heuristic-based and may fail on highly noisy, unusual, or non-invoice document layouts.
- The name extraction includes a small known-name bias and regex heuristics, so unseen formats may reduce accuracy.
- OCR quality depends heavily on image clarity, skew, lighting, and text size.
- The API currently extracts only three fields: `name`, `date`, and `amount`.
- Error handling returns generic API errors for unexpected processing failures and does not yet include detailed validation diagnostics.
- Evaluation uses exact normalized matching, so partially-correct outputs may still count as failures.

## Project Structure

```text
api/
  app.py
data/
  labels.csv
evaluate.py
outputs/
  evaluation_summary.json
  sample_results.json
src/
  extract.py
  ocr.py
  utils.py
requirements.txt
README.md
```

## Notes

- If PaddleOCR model initialization takes time on first run, allow the process to finish before testing requests.
- Keep the virtual environment activated while running the API and installing packages.
- Run [`evaluate.py`](evaluate.py:1) after testing to generate a reproducible accuracy summary for submission.
