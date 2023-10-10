import re
from typing import Dict, Union
from pypdf import PdfReader
from abc import ABC, abstractmethod
import pandas as pd

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
    def breakdown_document(document, reader: PdfReader = None, max_tokens: int = 1000, only_alphaNumeric: bool = False, bookmarks_to_ignore: set = set(), strip_bookmarks: set = set(), list_of_bookmarks = {}) -> Dict[Union[str, int], str]:
        if isinstance(document, str):
            pdfReader = PdfReader(document).outline
            reader = PdfReader(document)
            list_of_bookmarks = PDFParser.list_bookmarks(document)
        else:
            pdfReader = document
        
        result = {}
        bookmarks = list(pdfReader)
        for i in range(len(bookmarks)):
            item = bookmarks[i]
            if isinstance(item, list):
                # Recursive call with updated strip_bookmarks set
                result.update(PDFParser.breakdown_document(document=item, reader=reader, bookmarks_to_ignore=bookmarks_to_ignore, strip_bookmarks=strip_bookmarks, list_of_bookmarks=list_of_bookmarks))
            else:
                page_index = reader.get_destination_page_number(item)
                bookmark_name = item.title
                if(bookmark_name.strip().lower() not in {bookmark.lower() for bookmark in bookmarks_to_ignore}): 
                    
                    # Check if the bookmark is part of the strip_bookmarks set
                    if(strip_bookmarks == set() or 
                    bookmark_name.strip().lower() in {bookmark.lower() for bookmark in strip_bookmarks}):
                        print("Not skipping: " + bookmark_name)
                        # If the bookmark is not skipped, add its sub-bookmarks to the strip_bookmarks set
                        if strip_bookmarks != set() and i + 1 < len(bookmarks) and isinstance(bookmarks[i + 1], list):
                            strip_bookmarks = strip_bookmarks.union({sub_bookmark.title for sub_bookmark in bookmarks[i + 1]})
                    else:
                        continue
                
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
                    for page in reader.pages[page_index:next_page_index+1]:
                        page_text = page.extract_text()
                        # Locate the bookmark name in the page text
                        bookmark_start = page_text.find(bookmark_name)
                        if bookmark_start != -1:
                            # Extract the text from the bookmark name onwards
                            page_text = page_text[bookmark_start + len(bookmark_name):]
                        # Locate the next bookmark name in the page text
                        next_bookmark_name = ""
                        name = [item for item in list_of_bookmarks if item['/Title'] in [bookmark_name]]
                        if list_of_bookmarks.index(name[0]) < len(list_of_bookmarks) - 1:
                            next_bookmark = list_of_bookmarks[list_of_bookmarks.index(name[0]) + 1]
                            if isinstance(next_bookmark, dict):
                                next_bookmark_name = next_bookmark['/Title']  
                            elif isinstance(next_bookmark, list) and len(next_bookmark) > 0 and isinstance(next_bookmark[0], dict):
                                next_bookmark_name = next_bookmark[0]['/Title']
                              
                        if next_bookmark_name:
                            next_bookmark_start = page_text.find(next_bookmark_name)
                            if next_bookmark_start != -1:
                                # Extract the text up to the next bookmark name
                                page_text = page_text[:next_bookmark_start]
                        # Remove non-alphabetic characters
                        if only_alphaNumeric:
                            page_text = re.sub(r'\W+', ' ', page_text)
                        bookmark_content += page_text
                        print("adding: " + bookmark_name)
                        if next_bookmark_name not in strip_bookmarks and next_bookmark_start != -1:
                            break
                        
                    # Split the content into chunks of max_tokens
                    tokens = bookmark_content.split()
                    
                    for j in range(0, len(tokens), max_tokens):
                        chunk = " ".join(tokens[j:j+max_tokens])
                        #print('Adding: ' + f"{bookmark_name}_{j//max_tokens}")
                        result[f"{bookmark_name}_{j//max_tokens}"] = chunk 
        return result
    
    def flatten(lst):
        """Flattens a list of lists and/or nested lists to a single list"""
        for x in lst:
            if isinstance(x, list):
                for y in PDFParser.flatten(x):
                    yield y
            else:
                yield x

    def list_bookmarks(pdf_path):
        """Lists all bookmarks including sub-bookmarks from a PDF"""
        # Open the PDF
        pdf = PdfReader(pdf_path)
        
        # Get the outlines (bookmarks) of the PDF
        outlines = pdf.outline
        
        # Flatten the nested list of bookmarks to a single list
        bookmarks = list(PDFParser.flatten(outlines))
        
        # Return the list of bookmarks
        return bookmarks
    
    def save_to_csv(data_dict: dict, file_name: str):
        csv_file = file_name

        (pd.DataFrame.from_dict(data=data_dict, orient='index').to_csv(csv_file, header=False))

'''Example Usage'''
'''bms =  PDFParser.breakdown_document("RAP.pdf", 
                                    max_tokens=4000, only_alphaNumeric=False, 
                                    strip_bookmarks={'Reasoning via Planning (RAP)'})
                                    #bookmarks_to_ignore={'Answers to Odd-Numbered Exercises', 'End-of-Chapter Material'})
PDFParser.save_to_csv(bms, "RAP.csv")'''

'''print(bms)

for bm in bms:
    print(bm)
    print(bms[bm])'''