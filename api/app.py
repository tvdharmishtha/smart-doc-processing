"""
FastAPI Application for Smart Document Processing
Provides an endpoint to process document images and extract structured information.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import io
import pandas as pd

# Import our modules
from src.ocr import OCRPipeline, load_image_from_bytes
from src.extract import InformationExtractor
from src.utils import format_response

# Initialize the app
app = FastAPI(
    title="Smart Document Processing API",
    description="API for extracting structured information from document images",
    version="1.0.0"
)

# Initialize OCR and extraction components
ocr_pipeline = None
extractor = None


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


def format_structured_invoice_text(fields: Dict[str, Any]) -> str:
    """Build canonical normalized text output from extracted invoice fields."""
    lines = ["Invoice"]
    if fields.get("name"):
        lines.append(f"Name: {fields['name']}")
    if fields.get("date"):
        lines.append(f"Date: {fields['date']}")
    if fields.get("amount"):
        amount = str(fields["amount"])
        if amount.endswith(".00"):
            amount = amount[:-3]
        lines.append(f"Total: {amount}")
    return "\n".join(lines)


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
        # Read file contents
        contents = await file.read()
        
        # Check file type by content and filename
        filename = file.filename.lower() if file.filename else ""
        content_type = file.content_type if file.content_type else ""
        
        extracted_text = ""
        fields = {"name": None, "date": None, "amount": None}
        
        # Handle Excel files
        if filename.endswith(('.xlsx', '.xls', '.csv')) or 'spreadsheet' in content_type or 'excel' in content_type:
            try:
                # Read Excel/CSV file
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(contents))
                else:
                    df = pd.read_excel(io.BytesIO(contents))
                
                # Convert all data to text
                extracted_text = df.to_csv(index=False)

                # Extract structured information from table rows for Excel/CSV files
                extractor = get_extractor()
                rows = df.where(pd.notna(df), None).to_dict(orient="records")
                fields = extractor.extract_from_rows(rows)

                # Convert arrays to list format for JSON response
                # Keep the arrays in the fields for Excel files
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading Excel file: {str(e)}"
                )
        else:
            # Handle image files
            # Load image
            image = load_image_from_bytes(contents)
            
            # Extract text using OCR - always use preprocessing for better results
            ocr = get_ocr_pipeline()
            extracted_text = ocr.extract_text(image)

            if not extracted_text:
                return format_response("", {"name": None, "date": None, "amount": None})
            
            # Extract structured information
            extractor = get_extractor()
            fields = extractor.extract_all(extracted_text)

            # Dynamically normalize the OCR text from the extracted fields.
            if any(fields.values()):
                extracted_text = format_structured_invoice_text(fields)

        # Format and return response
        return format_response(extracted_text, fields)
        
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
