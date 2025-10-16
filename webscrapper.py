from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
import re
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
def bs4_extractor(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        divs = soup.find_all("div", class_="col-md-12 col-sm-12")
        text = "\n\n".join(div.get_text(separator=" ", strip=True) for div in divs)
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()
    except Exception as e:
        print("Error parsing HTML:", e)
        return ""
def safe_metadata_extractor(raw_html, url, response):
    # Skip non-HTML content
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        print(f"Skipping non-HTML: {url} ({content_type})")
        return {"source": url, "content_type": content_type}

    # Use html5lib parser to handle malformed HTML
    soup = BeautifulSoup(raw_html, "html5lib")
    title = soup.title.string if soup.title else "No Title"
    return {"source": url, "title": title}
def content_type_filter(response):
    return response.headers.get("Content-Type", "").startswith("text/html")
loader = RecursiveUrlLoader(
    "https://online.gndu.ac.in/departments.aspx",
    max_depth=2,
    extractor=bs4_extractor,
    metadata_extractor=safe_metadata_extractor
)
# Wont be using recurive loader, My laptop is on fucking fireee
# loader = WebBaseLoader(
#     "https://online.gndu.ac.in"
# )
docs = loader.load()
splitter=CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=150,
    separator=''
)
result=splitter.split_documents(docs)
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
# collection.add(embeddings=embeddings,documents=texts,ids=ids)
print(docs)