import os
import ssl
import asyncio
import certifi

from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from logger import log_error, log_header, log_info, log_success, log_warning
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUEST_CA_BUNDLE"]=certifi.where()


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore= PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)
tavily_extract=TavilyExtract()
tavily_map=TavilyMap(max_depth=5,max_breadth=20,max_pages=1000)
tavily_crawl=TavilyCrawl()

print("All are ok since now!")

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info("Starting tavilycrawl doc", Colors.PURPLE)

    res=tavily_crawl.invoke({
        "url":"https://python.langchain.com/",
        "max_depth":5,
        "extract_depth":"advanced",
        "instructions":"content on ai agents"
    })
    all_docs = []
    for result in res["results"]:
        content = result.get("raw_content")
        if not isinstance(content, str) or not content.strip():
            log_warning(f"Skipping page with no usable text: {result.get('url', 'unknown URL')}")
            continue
        all_docs.append(
            Document(page_content=content, metadata={"source": result["url"]})
        )

    log_success(
            f"TavilyCrawl: Created {len(all_docs)} documents from {len(res['results'])} returned pages"
        )

    log_header("Splitting docs into chunks")
    for doc in all_docs:
        log_info(f"{len(doc.page_content)} characters: {doc.metadata['source']}")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
    splitted_docs=text_splitter.split_documents(all_docs)
    log_success(
        f"Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

if __name__ == "__main__":
    asyncio.run(main())
