from langchain_core.prompts import PromptTemplate
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chat_history=[SystemMessage(content='You are helpful AI assistant')]
prompt=PromptTemplate(
    template=' Give me information about {thing}',
    input_variables=['thing']
)
prompt1 = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Conversation so far:\n{context}\n\n"
        "Relevant university data:\n{data}\n\n"
        "User query:\n{query}\n\n"
        "Now give a helpful, concise, and factual answer."
    ),
    input_variables=["context", "data", "query"]
)
parser=StrOutputParser()
chain=prompt1|model|parser
client = chromadb.PersistentClient(path="chroma_db")
data_collection=client.get_collection('Uni_collection')
while True:
    user_input=input('you: ')
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
    embedded_query=embedding_model.encode(user_input).tolist()
    data=data_collection.query(query_embeddings=embedded_query)
    data_docs=data["documents"][0]
    text="".join(data_docs)
    context_text = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
    if user_input=='exit':
        break
    result=chain.invoke({'context':context_text,'query':user_input,'data':text})
    chat_history.append(HumanMessage(content=prompt.format(thing=user_input)))
    chat_history.append(AIMessage(content=result))
    print("AI: ",result)
print(chat_history)