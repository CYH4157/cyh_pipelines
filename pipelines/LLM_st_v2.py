# coding: utf-8
import os
import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
import pandas as pd
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
import json

# 初始化嵌入模型
embed_model = OllamaEmbedding(
    model_name="chatfire/bge-m3:q8_0",
    base_url="http://172.17.0.1:11434", 
    ollama_additional_kwargs={"mirostat": 0}
)

# 初始化 Qdrant 客戶端並載入向量存儲
client = qdrant_client.QdrantClient(url="http://172.17.0.1:6333")
vector_store = QdrantVectorStore(client=client, collection_name="20240906_ly_256")
index = VectorStoreIndex.from_vector_store(embed_model=embed_model, vector_store=vector_store)

# 初始化重排序器
reranker = FlagEmbeddingReranker(
    top_n=2,
    model="BAAI/bge-reranker-large"
)

# 初始化 LLM
llm = Ollama(
    model='llama3.1:8b-instruct-fp16', 
    base_url="http://172.17.0.1:11434", 
    request_timeout=120.0
)

# 設置檢索引擎
retriever_engine = index.as_retriever(
    retriever_mode='embeddings',
    similarity_top_k=2,
    node_postprocessors=[reranker],
    verbose=True
)


# 初始化聊天信息
messages = [
    {"role": "assistant", "content": "Hi, ask me a question. My knowledge is always up to date!"}
]

# 獲取用戶輸入
prompt = input("Your question: ")
messages.append({"role": "user", "content": prompt})

# 處理檢索和生成回應
if messages[-1]["role"] != "assistant":
    print("Thinking...")
    retrieve_res = retriever_engine.retrieve(prompt)
    context_str = retrieve_res[0].node.excluded_embed_metadata_keys[0]
    query_str = prompt
    ques_str = (
        f"我提供的上下文內容如下：\n"
        f"---------------------\n"
        f"{context_str}\n"
        f"---------------------\n"
        f"基於給出的內容，回答下列問題: {query_str}\n"
    )
    response = llm.complete(ques_str)
    
    # 顯示檢索結果和模型回應
    print("Retrieve Results:", retrieve_res)
    print("Response:", response.text)
    message = {"role": "assistant", "content": response.text}
    messages.append(message)


