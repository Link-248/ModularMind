import re
from typing import Dict, Union
from pypdf import PdfReader
from abc import ABC, abstractmethod

class VectorStoreBase(ABC):
    
    @abstractmethod
    def __init__(self):
        pass