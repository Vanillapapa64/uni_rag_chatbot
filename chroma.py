import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()
loader = PyPDFLoader('Prospectus202526.pdf')
docs=loader.load()
splitter=CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=150,
    separator=''
)
result=splitter.split_documents(docs)
texts = [doc.page_content for doc in result if doc.page_content.strip()]
client = chromadb.PersistentClient(path="chroma_db")
collection_name = "Uni_collection"
if collection_name not in [c.name for c in client.list_collections()]:
    collection = client.create_collection(name=collection_name)
else:
    collection = client.get_collection(name=collection_name)
# embedding_model=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001',task_type="RETRIEVAL_DOCUMENT")
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc.page_content for doc in result if doc.page_content.strip()]
ids = [f"doc{i}" for i in range(len(texts))]
embeddings=embedding_model.encode(texts).tolist()
collection.add(embeddings=embeddings,documents=texts,ids=ids)

# def add(docs,ids):

