import re
from typing import Dict, Union
from pypdf import PdfReader
from abc import ABC, abstractmethod

class DocumentParserBase(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def breakdown_document(self):
        pass
    
class PDFParser(DocumentParserBase):
    """
    Extract all bookmarks as a flat dictionary and links them to their content. content is split by token limit

    Args:
        document: The reader.outline or str of the document. Used in a recursive call
        reader: The PdfReader object. Used in a recursive call
        max_tokens: The maximum number of tokens to include in each bookmark section chunk.

    Returns:
        A dictionary mapping PDF bookmark sections to their content

    Examples:
        Download the PDF from https://zenodo.org/record/50395 to give it a try
    """
    def breakdown_document(document, reader: PdfReader = None, max_tokens: int = 1000, only_alphaNumeric: bool = False) -> Dict[Union[str, int], str]:
        if isinstance(document, str):
            pdfReader = PdfReader(document).outline
            reader = PdfReader(document)
        else:
            pdfReader = document
           
        result = {}
        bookmarks = list(pdfReader)
        for i in range(len(bookmarks)):
            item = bookmarks[i]
            if isinstance(item, list):
                #recursive call
                result.update(PDFParser.breakdown_document(item, reader))
            else:
                page_index = reader.get_destination_page_number(item)
                bookmark_name = item.title
                    
                # Get the page number of the next bookmark
                if i + 1 < len(bookmarks):
                    next_item = bookmarks[i + 1]
                    if isinstance(next_item, list):
                        if next_item:
                            next_page_index = reader.get_destination_page_number(next_item[0])
                        else:
                            continue
                    else:
                        next_page_index = reader.get_destination_page_number(next_item)
                else:
                    # If this is the last bookmark, get the number of the last page in the PDF
                    next_page_index = len(reader.pages)

                # Extract all pages from the current bookmark up to (but not including) the next bookmark
                bookmark_content = ""
                for page in reader.pages[page_index:next_page_index]:
                    page_text = page.extract_text()
                    # Remove non-alphabetic characters
                    if only_alphaNumeric:
                        page_text = re.sub(r'\W+', ' ', page_text)
                    bookmark_content += page_text

                # Split the content into chunks of max_tokens
                tokens = bookmark_content.split()
                for j in range(0, len(tokens), max_tokens):
                    chunk = " ".join(tokens[j:j+max_tokens])
                    result[f"{bookmark_name}_{j//max_tokens}"] = chunk          

        return result

'''Example Usage
bms =  PDFParser.breakdown_document("benefits-booklet.pdf", max_tokens=4000, only_alphaNumeric=False)
print(bms)

for bm in bms:
    print(bm)
    print(bms[bm])'''