import os, json

import faiss
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.schema import TextNode, Node
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever

from ..utils.optimum_embeddings import OptimumEmbedding
from ..utils.processing import convert_cnbc_to_fulltext, convert_world_news_to_fulltext, generate_article_files

class VectorStoreRetriever:
    config_rt_to_ep = {
        "TRT": "TensorrtExecutionProvider",
        "CUDA": "CUDAExecutionProvider",
        "CPU": "CPUExecutionProvider"
    }
    def generate_documents(self):
        if self._config["faiss-dataset"]=='cnbc':
            df = convert_cnbc_to_fulltext(f"{self._raw_data_dir}\crawlfeeds-cnbc-news-dataset\original\cnbc_news_datase.csv")
        elif self._config["faiss-dataset"]=="world-news":
            df = convert_world_news_to_fulltext(f"{self._raw_data_dir}\global-news-dataset\data.csv")
        generate_article_files(df, self._proccesed_data_dir)

    def generate_index(self):
        documents = SimpleDirectoryReader(input_dir=self._proccesed_data_dir, recursive=True).load_data()
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self._embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)
        self._faiss_index = faiss.IndexFlatL2(self._embed_model.emb_size)
        self._vector_store = FaissVectorStore(faiss_index=self._faiss_index)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex(nodes=nodes, embed_model=self._embed_model, storage_context=self._storage_context)
        self._index.storage_context.persist(persist_dir=f'{self._faiss_data_dir}/faiss-storage-context')

    def __init__(self, data_dir: str='../data', config: str='../config.json'):
        self._raw_data_dir = os.path.join(data_dir, 'raw')
        self._proccesed_data_dir = os.path.join(data_dir, 'processed')
        self._faiss_data_dir = os.path.join(data_dir, 'faiss')
        self._embed_model = OptimumEmbedding(
            folder_name=self._config["hf-model-id"],
            provider=self.config_rt_to_ep[config["onnx-runtime"]]
        )

        with open(config, 'r', encoding='utf8') as f:
            self._config = json.load(f)
        
        if self._config["faiss-regenerate-on-start"] or not len(self._proccesed_data_dir):
            self.generate_documents()
            self.generate_index()
        else:
            vector_store = FaissVectorStore.from_persist_dir(f'{self._faiss_data_dir}/faiss-storage-context')
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=f'{self._faiss_data_dir}/faiss-storage-context'
            )
            self._index = load_index_from_storage(storage_context=storage_context)
        self._retriever = VectorIndexRetriever(index=self._index, similarity_top_k=5, embed_model=self._embed_model)

    def retrieve(self, text) -> list[Node]:
        return self._retriever.retrieve(text)

# # load index from disk
# vector_store = FaissVectorStore.from_persist_dir("./storage")
# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store, persist_dir="./storage"
# )
# index = load_index_from_storage(storage_context=storage_context)