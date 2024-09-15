"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
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
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
import nltk
class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        Qdrant_BASE_URL: str
        Qdrant_VectorStore: str
        Flag_Embedding_Reranker: str

    def __init__(self):
        self.type = "pipe"
        self.id = "cyh_lamaindex_pl"
        self.name = "cyh_lamaindex_pl"
        #self.documents = None
        #self.index = None

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
        print('=============  ollama setting =============')
        # Settings.embed_model = OllamaEmbedding(
        #     model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
        #     base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        # )

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            ollama_additional_kwargs={"mirostat": 0}
        )        

        Settings.client = qdrant_client.QdrantClient(url=self.valves.Qdrant_BASE_URL)

        Settings.vector_store = QdrantVectorStore(client=Settings.client, collection_name=self.valves.Qdrant_VectorStore)

        # inital Reranker
        Settings.reranker = FlagEmbeddingReranker(
            top_n=2,
            model=self.valves.Flag_Embedding_Reranker
        )


        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            request_timeout=120.0
        )
        


        print('=============  ollama setting finished =============')
        #documents = SimpleDirectoryReader("./data").load_data()
        #index = VectorStoreIndex.from_documents(documents)

        index = VectorStoreIndex.from_vector_store(embed_model=Settings.embed_model, vector_store=Settings.vector_store)

        retriever_engine = index.as_retriever(
            retriever_mode='embeddings',
            similarity_top_k=2,
            node_postprocessors=[reranker],
            verbose=True
        )


        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)


        retrieve_res = retriever_engine.retrieve(user_message)
        context_str = retrieve_res[0].node.excluded_embed_metadata_keys[0] if retrieve_res else "No relevant context found."
        ques_str = (
            f"我提供的上下文內容如下：\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n"
            f"基於給出的內容，回答下列問題: {user_message}\n"
        )

        # Display search results and model responses
        response = llm.complete(ques_str)

        print("Retrieve Results:", retrieve_res)
        print("Response:", response.text)


        return response.response_gen
