from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader('Prospectus202526.pdf')
from PyPDF2 import PdfReader
docs=loader.load()
splitter=CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=150,
    separator=''
)
result=splitter.split_documents(docs)
for i in range(0,len(result)):
    print(result[i].page_content)
