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
from pathlib import Path
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
        self.id = "nchc-beta-lamaindex_test_4"
        self.name = "nchc-beta-lamaindex_test_4"
        
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1:8b-instruct-fp16"),
            }
        )

    async def on_startup(self):

        # This function is called when the server is started.
        # global documents, index

        # self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        # self.index = VectorStoreIndex.from_documents(self.documents)



        Settings.embed_model = OllamaEmbedding(
            model_name="chatfire/bge-m3:q8_0",
            base_url="http://ollama:11434",
            ollama_additional_kwargs={"mirostat": 0}
        )        

        Settings.client = qdrant_client.QdrantClient(url="http://qdrant:6333")

        Settings.vector_store = QdrantVectorStore(client=Settings.client, collection_name="20240906_ly_256")

        print('======== Reranker ================')
        # inital Reranker
        reranker = FlagEmbeddingReranker(
            top_n=2,
            model="BAAI/bge-reranker-large"
        )
        print('======== Reranker ================')

        self.llm = Ollama(
            model='llama3.1:8b-instruct-fp16',
            base_url="http://ollama:11434"
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
        
        system_content = [msg['content'] for msg in messages if msg['role'] == 'system']
        
        if system_content:
            print("找到 'system' 的部分：")
            
            print(os.getcwd())

            directory = "/app/backend/data/uploads"
            if not os.path.exists(directory):
                print(f"The directory {directory} does not exist.")
            pdf_files = list(Path(directory).rglob("*.pdf"))
            print (pdf_files)
            latest_pdf = max(pdf_files, key=lambda f: f.stat().st_mtime)
            
            context_str = pymupdf4llm.to_markdown(latest_pdf)
                        
            ques_str = (
                f"我提供的上下文內容如下：\n"
                f"---------------------\n"
                f"{context_str}\n"
                f"---------------------\n"
                f"基於給出的內容，回答下列問題: {user_message}\n"
            )

            try:
                response = self.llm.complete(ques_str)
            except Exception as e:
                print(f"Failed to response: {e}")
                
            final_res = response.text
           

        else:
            print("這次回傳沒有 'system' 的部分")

            retrieve_res = self.retriever_engine.retrieve(user_message)

    
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
    
    
            try:
                response = self.llm.complete(ques_str)
            except Exception as e:
                print(f"Failed to response: {e}")

            final_res = response.text

        #print("Retrieve Results:", retrieve_res)
        #print("Response:", response.text)
        #for message in messages:
        #    print(f"Message keys: {message.keys()}")
        
        #print('body:', body)


        return final_res
