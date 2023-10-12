import pinecone
from tqdm.auto import tqdm  # this is the pinecone progress bar
import re
from typing import Union, List, Tuple, Optional, Dict
from pinecone.core.client.models import QueryVector
from pinecone.core.client.model.sparse_values import SparseValues
   
#Pinecone vector store wrapper
class Pinecone():
    index: pinecone.Index
    
    def __init__(self, api_key: str, index_name: str, environment: str = 'gcp-starter', dimension: int = 1536):
        # set index name to all lower cases and change any non-alphanumeric characters to dashes
        index_name = index_name.lower()
        index_name = re.sub('[^0-9a-zA-Z]+', '-', index_name)
        
        # initialize connection to pinecone (get API key at app.pinecone.io)
        pinecone.init(
            api_key=api_key,
            environment=environment  # find next to API key in console
        )    
        # check if index already exists (only create index if not)
        if index_name not in pinecone.list_indexes():
            self.index = pinecone.create_index(index_name, dimension=dimension)
        # connect to index
        self.index = pinecone.Index(index_name)
    
    # upsert embeddings from a dictionary of {content: embeddings} key value pairs
    def upsert_embeddings_from_dict(self, dict: dict, metadata_name: str = 'content'):
        batch_size = 32  # process everything in batches of 32
        for i in tqdm(range(0, len(list(dict.keys())), batch_size)):
            # set end position of batch
            i_end = min(i+batch_size, len(list(dict.keys())))
            # get batch of lines and IDs
            lines_batch = list(dict.keys())[i: i+batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]
            
            
            # store embeddings
            embeds = list(dict.values())
            # prep metadata and upsert batch
            meta = [{metadata_name: line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            self.index.upsert(vectors=list(to_upsert))
    
    def query(self,
              vector: Optional[List[float]] = None,
              id: Optional[str] = None,
              queries: Optional[Union[List[QueryVector], List[Tuple]]] = None,
              top_k: Optional[int] = None,
              namespace: Optional[str] = None,
              filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
              include_values: Optional[bool] = None,
              include_metadata: Optional[bool] = None,
              sparse_vector: Optional[Union[SparseValues, Dict[str, Union[List[float], List[int]]]]] = None,
              **kwargs):
        return self.index.query(vector=vector, id=id, queries=queries, top_k=top_k, namespace=namespace, filter=filter, include_values=include_values, include_metadata=include_metadata, sparse_vector=sparse_vector, **kwargs)
