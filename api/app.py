import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import io
import pandas as pd

from src.ocr import OCRPipeline, load_image_from_bytes
from src.extract import InformationExtractor
from src.utils import format_response, build_result_record, append_unique_result

app = FastAPI(
    title="Smart Document Processing API",
    description="API for extracting structured information from document images",
    version="1.0.0"
)

ocr_pipeline = None
extractor = None
sample_results_path = project_root / "outputs" / "sample_results.json"


def get_ocr_pipeline():
    """Get or create OCR pipeline instance."""
    global ocr_pipeline
    if ocr_pipeline is None:
        ocr_pipeline = OCRPipeline(languages=['en'], use_gpu=False)
    return ocr_pipeline


def get_extractor():
    """Get or create information extractor instance."""
    global extractor
    if extractor is None:
        extractor = InformationExtractor()
    return extractor


class ProcessDocumentResponse(BaseModel):
    """Response model for document processing."""
    extracted_text: str
    fields: Dict[str, Union[str, List[str], None]]


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    get_ocr_pipeline()
    get_extractor()


@app.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process a document (image or Excel) and extract structured information.
    
    Args:
        file: Uploaded file (image or Excel)
        
    Returns:
        JSON response with extracted text and fields
    """
    try:
        contents = await file.read()

        original_filename = file.filename or ""
        filename = original_filename.lower()
        content_type = file.content_type if file.content_type else ""

        extracted_text = ""
        fields = {"name": None, "date": None, "amount": None}

        if filename.endswith(('.xlsx', '.xls', '.csv')) or 'spreadsheet' in content_type or 'excel' in content_type:
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(contents))
                else:
                    df = pd.read_excel(io.BytesIO(contents))

                extracted_text = df.to_csv(index=False)

                extractor = get_extractor()
                rows = df.where(pd.notna(df), None).to_dict(orient="records")
                fields = extractor.extract_from_rows(rows)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading Excel file: {str(e)}"
                )
        else:
            image = load_image_from_bytes(contents)

            ocr = get_ocr_pipeline()
            extracted_text = ocr.extract_text(image)

            if not extracted_text:
                response_payload = format_response("", {"name": None, "date": None, "amount": None})
                result_record = build_result_record(original_filename, contents, response_payload, content_type)
                append_unique_result(result_record, str(sample_results_path))
                return response_payload

            extractor = get_extractor()
            fields = extractor.extract_all(extracted_text)

        response_payload = format_response(extracted_text, fields)
        result_record = build_result_record(original_filename, contents, response_payload, content_type)
        append_unique_result(result_record, str(sample_results_path))

        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
