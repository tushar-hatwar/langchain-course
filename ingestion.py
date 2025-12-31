import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from azure_env_embed import embeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "mediumblog1.txt"), encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("ingesting...")
    print(os.environ["INDEX_NAME"])
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
