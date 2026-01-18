import logging
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extrai o texto de um arquivo PDF.
    
    Args:
        file_path: Caminho do arquivo PDF
        
    Returns:
        Texto extraído do PDF
        
    Raises:
        FileNotFoundError: Se o arquivo não existe
        ValueError: Se não é um arquivo PDF válido
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Arquivo não é um PDF: {file_path}")
    
    try:
        reader = PdfReader(file_path)
        
        if reader.is_encrypted:
            logger.warning(f"PDF criptografado: {file_path}")
            
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            logger.warning(f"Nenhum texto encontrado no PDF: {file_path}")
            
        return text.strip()
        
    except Exception as e:
        logger.error(f"Erro ao processar PDF {file_path}: {str(e)}")
        raise


def extract_text_from_pdf_by_page(file_path: str) -> List[Dict[str, str]]:
    """
    Extrai texto de um PDF, separando por página.
    
    Args:
        file_path: Caminho do arquivo PDF
        
    Returns:
        Lista de dicionários com número da página e texto
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    try:
        reader = PdfReader(file_path)
        pages_data = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                pages_data.append({
                    "page": page_num,
                    "text": text
                })
        
        return pages_data
        
    except Exception as e:
        logger.error(f"Erro ao processar PDF por página {file_path}: {str(e)}")
        raise


def extract_text_from_multiple_pdfs(directory: str) -> Dict[str, str]:
    """
    Processa múltiplos PDFs de um diretório.
    
    Args:
        directory: Caminho do diretório com PDFs
        
    Returns:
        Dicionário com {nome_arquivo: texto_extraído}
    """
    dir_path = Path(directory)
    
    if not dir_path.is_dir():
        raise ValueError(f"Diretório não encontrado: {directory}")
    
    results = {}
    pdf_files = list(dir_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"Nenhum PDF encontrado em: {directory}")
    
    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(str(pdf_file))
            results[pdf_file.name] = text
            logger.info(f"✓ Processado: {pdf_file.name}")
        except Exception as e:
            logger.error(f"✗ Erro em {pdf_file.name}: {str(e)}")
    
    return results
