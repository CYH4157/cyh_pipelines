"""
title: nchc-beta Llama Index Ollama Pipeline
author: nchc-beta
date: 2024-09-20
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.

"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, StorageContext



import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
import pandas as pd
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
import json


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_MODEL_NAME: str

    def __init__(self):
        self.llm = None
        self.id = "nchc-beta-lamaindex"
        self.name = "nchc-beta-lamaindex"
        
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://172.17.0.1:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1:8b-instruct-fp16"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "chatfire/bge-m3:q8_0"),
                "Qdrant_BASE_URL": os.getenv("Qdrant_BASE_URL", "http://172.17.0.1:6333"),
                "Qdrant_VectorStore": os.getenv("Qdrant_VectorStore", "20240906_ly_256"),
                "Flag_Embedding_Reranker": os.getenv("Flag_Embedding_Reranker", "BAAI/bge-reranker-large"),
                

            }
        )

    async def on_startup(self):

        # This function is called when the server is started.
        # global documents, index

        # self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        # self.index = VectorStoreIndex.from_documents(self.documents)



        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            ollama_additional_kwargs={"mirostat": 0}
        )        

        Settings.client = qdrant_client.QdrantClient(url=self.valves.Qdrant_BASE_URL)

        Settings.vector_store = QdrantVectorStore(client=Settings.client, collection_name=self.valves.Qdrant_VectorStore)

        print('======== Reranker ================')
        # inital Reranker
        reranker = FlagEmbeddingReranker(
            top_n=2,
            model=self.valves.Flag_Embedding_Reranker
        )
        print('======== Reranker ================')

        self.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL
        )
        


        print('=============  ollama setting finished =============')


        index = VectorStoreIndex.from_vector_store(embed_model=Settings.embed_model, vector_store=Settings.vector_store)

        self.retriever_engine = index.as_retriever(
            retriever_mode='embeddings',
            similarity_top_k=2,
            node_postprocessors=[reranker],
            verbose=True
        )



        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)
        # print('=============  ollama setting =============')
        # Settings.embed_model = OllamaEmbedding(
        #     model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
        #     base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        # )


        #documents = SimpleDirectoryReader("./data").load_data()
        #index = VectorStoreIndex.from_documents(documents)

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)


        retrieve_res = self.retriever_engine.retrieve(user_message)
        #context_str = retrieve_res[0].node.excluded_embed_metadata_keys[0] if retrieve_res else "No relevant context found."

        try:
            context_str = retrieve_res[0].node.excluded_embed_metadata_keys[0] if retrieve_res else "No relevant context found."
        except Exception as e:
            print(f"Failed to context_str: {e}")

        ques_str = (
            f"我提供的上下文內容如下：\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n"
            f"基於給出的內容，回答下列問題: {user_message}\n"
        )

        # Display search results and model responses
        #response = llm.complete(ques_str)

        try:
            response = self.llm.complete(ques_str)
        except Exception as e:
            print(f"Failed to response: {e}")

        print("Retrieve Results:", retrieve_res)
        print("Response:", response.text)


        return response.text
