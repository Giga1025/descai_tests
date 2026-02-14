# Add_data

# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import glob, argparse
import json
from pathlib import Path
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from transformers import AutoTokenizer
#----
from docling.datamodel.base_models import InputFormat

# === CONFIG ===
EMBED_MODEL_ID = "BAAI/bge-m3"
MAX_TOKENS = 300
DEFAULT_OUTPUT_JSONL = "text_knowledge_base.jsonl"
IMAGE_RESOLUTION_SCALE = 2.0

# === SETUP ===
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = False

pdf_format_options = PdfFormatOption(pipeline_options=pipeline_options)

converter = DocumentConverter(
    format_options={"pdf": pdf_format_options}
)

# ============================Image Export Converter==============================
def get_image_export_converter():
    """Get a converter configured for image export."""
    img_pipeline_options = PdfPipelineOptions()
    img_pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    img_pipeline_options.generate_page_images = True
    img_pipeline_options.generate_picture_images = True
    img_pipeline_options.do_table_structure = True
    
    img_pdf_format_options = PdfFormatOption(pipeline_options=img_pipeline_options)
    
    return DocumentConverter(
        format_options={"pdf": img_pdf_format_options}
    )

# ============================Export figures and tables==============================
def export_figures(pdf_path, output_base_dir="exported_figures"):
    """Export figures and tables from a PDF."""
    doc_name = os.path.basename(pdf_path).replace(".pdf", "")
    output_dir = Path(output_base_dir) / doc_name
    
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Gettingt the figures from{pdf_path}...")
    
    img_converter = get_image_export_converter()
    conv_res = img_converter.convert(pdf_path)
    
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = tables_dir / f"{doc_name}-table-{table_counter}.png"
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
        
        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = images_dir / f"{doc_name}-picture-{picture_counter}.png"
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
    
    print(f"Exported {picture_counter} images and {table_counter} tables to {output_dir}")
    return output_dir

# ============================Load existing chunks to get last chunk_id==============================
def get_last_chunk_id(jsonl_path):
    if not os.path.exists(jsonl_path):
        return 0
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        if not lines:
            return 0
        last = json.loads(lines[-1])
        return last["chunk_id"] + 1
    
def get_last_chunk_id_fallback(jsonl_path):
    if os.path.exists(jsonl_path):
        return get_last_chunk_id(jsonl_path)
    return 0

# ============================Find all PDFs recursively(this is used only when we have multiple subfolders with pdfs)==============================
def find_all_pdfs(root_folder):
    """Recursively find all PDF files in subdirectories."""
    pdf_files = []
    root_path = Path(root_folder)
    
    # Use rglob to recursively find all PDFs
    for pdf_path in root_path.rglob("*.pdf"):
        pdf_files.append(pdf_path)
    
    return sorted(pdf_files)

# =================================== CHUNKING ALL PDFs using docling==============================
def process_folder(folder_path=None, file_path=None, starting_chunk_id=0, export_images=False, recursive=True):
    all_chunks = []
    chunk_id = starting_chunk_id

    if file_path:
        print(f"Processing single file: {file_path}")
        
        # Export figures if requested
        if export_images:
            export_figures(file_path)
        
        pdf_path_obj = Path(file_path)
        doc_name = pdf_path_obj.stem

        relative_path = pdf_path_obj.parent.name
        
        doc = converter.convert(source=file_path).document

        for chunk in chunker.chunk(dl_doc=doc):
            ser_txt = chunker.contextualize(chunk=chunk)
            token_count = tokenizer.count_tokens(ser_txt)

            heading, _, body = ser_txt.partition("\n")
            body = body.strip()

            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "category": relative_path,
                "section_heading": heading,
                "text": body,
                "token_count": token_count
            })

            chunk_id += 1
            
    elif folder_path:
        if recursive:
            # Process all PDFs in subdirectories
            pdf_files = find_all_pdfs(folder_path)
            total_files = len(pdf_files)
            print(f"Found {total_files} PDF files in subdirectories")
            
            for idx, pdf_path in enumerate(pdf_files, 1):
                pdf_path_str = str(pdf_path)
                print(f"[{idx}/{total_files}] Processing: {pdf_path_str}")
                
                #Gets imagges and tables if requested
                if export_images:
                    export_figures(pdf_path_str)
                
                doc_name = pdf_path.stem
                
                category = pdf_path.parent.name
                
                try:
                    doc = converter.convert(source=pdf_path_str).document

                    for chunk in chunker.chunk(dl_doc=doc):
                        ser_txt = chunker.contextualize(chunk=chunk)
                        token_count = tokenizer.count_tokens(ser_txt)

                        heading, _, body = ser_txt.partition("\n")
                        body = body.strip()

                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "doc_name": doc_name,
                            "category": category,
                            "section_heading": heading,
                            "text": body,
                            "token_count": token_count
                        })

                        chunk_id += 1
                except Exception as e:
                    print(f"Error processing {pdf_path_str}: {e}")
                    continue
        else:
            for pdf_path in glob.glob(os.path.join(folder_path, "*.pdf")):
                print(f"Processing: {pdf_path}")
                
                # Export figures if requested
                if export_images:
                    export_figures(pdf_path)
                
                doc_name = os.path.basename(pdf_path).replace(".pdf", "")
                doc = converter.convert(source=pdf_path).document

                for chunk in chunker.chunk(dl_doc=doc):
                    ser_txt = chunker.contextualize(chunk=chunk)
                    token_count = tokenizer.count_tokens(ser_txt)

                    heading, _, body = ser_txt.partition("\n")
                    body = body.strip()

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "doc_name": doc_name,
                        "category": "root",
                        "section_heading": heading,
                        "text": body,
                        "token_count": token_count
                    })

                    chunk_id += 1

    return all_chunks

# =================================================RUN ===================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and store PDF data into a JSONL knowledge base.")
    parser.add_argument("--folder", "-F", type=str, help="Path to the folder with PDFs.")
    parser.add_argument("--file", "-f", type=str, help="Path to a single PDF file to process.")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    parser.add_argument("--append", "-a", action="store_true", help="Append to existing JSONL and continue chunk IDs.")
    parser.add_argument("--export-images", "-e", action="store_true", help="Export images and tables from PDFs.")
    parser.add_argument("--no-recursive", "-nr", action="store_true", help="Don't process subdirectories recursively.")
    args = parser.parse_args()

    FOLDER_PATH = args.folder
    FILE_PATH = args.file
    OUTPUT_JSONL = args.output
    append_mode = args.append
    recursive = not args.no_recursive
    
    start_id = get_last_chunk_id_fallback(OUTPUT_JSONL) if append_mode else 0
    new_chunks = process_folder(FOLDER_PATH, FILE_PATH, start_id, export_images=args.export_images, recursive=recursive)

    if append_mode:
        with open("temporary_data.jsonl", "w") as temp_f:
            for chunk in new_chunks:
                temp_f.write(json.dumps(chunk) + "\n")

    write_mode = "a" if append_mode else "w"
    with open(OUTPUT_JSONL, write_mode) as f:
        for chunk in new_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n✓ {len(new_chunks)} chunks written to {OUTPUT_JSONL}")
    
    if new_chunks:
        from collections import Counter
        categories = Counter(chunk["category"] for chunk in new_chunks)
        print("\nChunks per category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
