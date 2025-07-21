import os
import re
import fitz
import pytesseract
import json
import logging
import tempfile
import subprocess
import argparse
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_office_to_pdf(input_path: str, output_dir: str) -> str:
    """Convert PowerPoint file to PDF using LibreOffice."""
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}.pdf")
    
    try:
        cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_path]
        result = subprocess.run(cmd, check=True, capture_output=True)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Conversion failed: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"LibreOffice error converting {input_path}: {e.stderr.decode()}")
        raise
    except Exception as e:
        logging.error(f"Failed to convert {input_path}: {str(e)}")
        raise

def extract_text_with_fallback(page: fitz.Page) -> Tuple[str, bool]:
    """Extract text from PDF page with OCR fallback."""
    text = page.get_text()
    if text.strip():
        return text, False
    
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text, True

def clean_text(text: str) -> str:
    """Clean extracted text from PDF artifacts."""
    # Remove page numbers and headers/footers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Rejoin hyphenated words
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leftover hyphen artifacts
    text = re.sub(r'\s?-\s?', '', text)
    return text.strip()

def infer_title(page_text: str) -> str:
    """Infer slide title from page text."""
    lines = page_text.split('\n')
    if lines:
        # Find the first line with substantial content
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                return stripped[:100]
    return "Untitled"

def process_pdf(file_path: str, chunk_size: int = 400, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Process single PDF into structured chunks."""
    doc_chunks = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text, used_ocr = extract_text_with_fallback(page)
            cleaned_text = clean_text(text)
            
            if not cleaned_text:
                continue
                
            title = infer_title(cleaned_text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(cleaned_text)
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": os.path.basename(file_path),
                    "page": page_num + 1,
                    "chunk_id": f"p{page_num+1}c{i+1}",
                    "title": title,
                    "ocr_used": used_ocr
                }
                doc_chunks.append({"page_content": chunk, "metadata": metadata})
                
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
    return doc_chunks

def process_directory(input_dir: str, chunk_size: int = 400, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Process all files in directory into structured chunks."""
    all_chunks = []
    if not os.path.exists(input_dir):
        logging.error(f"Directory not found: {input_dir}")
        return all_chunks

    # Create temp dir for converted PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process office files first
        for filename in os.listdir(input_dir):
            full_path = os.path.join(input_dir, filename)
            if os.path.isfile(full_path):
                if filename.lower().endswith(('.ppt', '.pptx')):
                    try:
                        logging.info(f"Converting: {filename}")
                        pdf_path = convert_office_to_pdf(full_path, temp_dir)
                        all_chunks.extend(process_pdf(pdf_path, chunk_size, chunk_overlap))
                    except Exception as e:
                        logging.error(f"Skipping {filename}: {str(e)}")
                elif filename.lower().endswith('.pdf'):
                    try:
                        logging.info(f"Processing: {filename}")
                        all_chunks.extend(process_pdf(full_path, chunk_size, chunk_overlap))
                    except Exception as e:
                        logging.error(f"Skipping {filename}: {str(e)}")
        
        # Process converted PDFs in temp dir
        for filename in os.listdir(temp_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(temp_dir, filename)
                logging.info(f"Processing converted: {filename}")
                all_chunks.extend(process_pdf(file_path, chunk_size, chunk_overlap))
                
    return all_chunks

def process_database_structure(database_dir: str, chunk_size: int = 400, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Process entire database structure with subdirectories."""
    all_chunks = []
    if not os.path.exists(database_dir):
        logging.error(f"Database directory not found: {database_dir}")
        return all_chunks

    # Process each subdirectory in the database
    for subdir in os.listdir(database_dir):
        subdir_path = os.path.join(database_dir, subdir)
        if os.path.isdir(subdir_path):
            logging.info(f"Processing subdirectory: {subdir}")
            chunks = process_directory(subdir_path, chunk_size, chunk_overlap)
            # Add course/subject information to metadata
            for chunk in chunks:
                chunk['metadata']['course'] = subdir
            all_chunks.extend(chunks)
            logging.info(f"Processed {len(chunks)} chunks from {subdir}")
    
    return all_chunks

def save_database(chunks: List[Dict[str, Any]], output_dir: str) -> None:
    """Save processed chunks to JSON files organized by source."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group chunks by course and then by source file
    course_groups = {}
    for chunk in chunks:
        course = chunk['metadata'].get('course', 'unknown')
        source = chunk['metadata']['source']
        
        if course not in course_groups:
            course_groups[course] = {}
        if source not in course_groups[course]:
            course_groups[course][source] = []
        
        course_groups[course][source].append(chunk)
    
    # Save each course as a separate JSON file
    for course, sources in course_groups.items():
        course_output_path = os.path.join(output_dir, f"{course}.json")
        course_data = []
        
        for source, source_chunks in sources.items():
            course_data.extend(source_chunks)
        
        try:
            with open(course_output_path, 'w', encoding='utf-8') as f:
                json.dump(course_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(course_data)} chunks from {course} to {course_output_path}")
        except Exception as e:
            logging.error(f"Failed to save {course}: {str(e)}")

def process_multiple_directories(input_dirs: List[str], output_dir: str, 
                                chunk_size: int = 400, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Process multiple input directories and save results."""
    all_chunks = []
    for input_dir in input_dirs:
        logging.info(f"Processing directory: {input_dir}")
        chunks = process_directory(input_dir, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        logging.info(f"Processed {len(chunks)} chunks from {input_dir}")
    
    if output_dir:
        save_database(all_chunks, output_dir)
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process lecture slides for embedding generation.')
    parser.add_argument('--input_dirs', nargs='+', 
                        help='List of input directories with PDF/PPT files')
    parser.add_argument('--database_dir', 
                        help='Database directory containing subdirectories with course materials')
    parser.add_argument('--output_dir', required=True, 
                        help='Output directory for saving JSON files')
    parser.add_argument('--chunk_size', type=int, default=400, 
                        help='Text chunk size in tokens')
    parser.add_argument('--chunk_overlap', type=int, default=100, 
                        help='Overlap between chunks')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process based on input type
    if args.database_dir:
        # Process entire database structure
        logging.info(f"Processing database structure: {args.database_dir}")
        result = process_database_structure(
            database_dir=args.database_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    elif args.input_dirs:
        # Process individual directories
        result = process_multiple_directories(
            input_dirs=args.input_dirs,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    else:
        logging.error("Either --input_dirs or --database_dir must be specified")
        exit(1)
    
    if args.database_dir:
        save_database(result, args.output_dir)
    
    logging.info(f"Total chunks processed: {len(result)}")
    logging.info(f"Database saved to: {args.output_dir}")