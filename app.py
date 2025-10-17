from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize your LangChain components
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chat_history = [SystemMessage(content='You are helpful AI assistant')]

prompt = PromptTemplate(
    template='Give me information about {thing}',
    input_variables=['thing']
)

prompt1 = PromptTemplate(
    template=(
        "You are a friendly and knowledgeable university assistant.\n"
        "Here’s the conversation so far:\n{context}\n\n"
        "Here’s the relevant university data:\n{data}\n\n"
        "User's latest queries:\n{query}\n\n"
        "Now, based on all of the above, answer naturally and clearly:\n"
        "- Avoid repeating the same sentence structure.\n"
        "- If you already mentioned a detail earlier, summarize instead of restating.\n"
        "- Give short, well-formatted bullet points when listing options.\n"
        "- Use conversational tone (e.g., 'You can also consider...', 'Another good choice is...').\n"
        "- Always end your response with a helpful next step or offer to help further."
    ),
    input_variables=["context", "data", "query"]
)


parser = StrOutputParser()
chain = prompt1 | model | parser

# Initialize ChromaDB and embedding model
client = chromadb.PersistentClient(path="chroma_db")
data_collection = client.get_collection('Uni_collection')
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    print('hello world')
    if request.method == 'OPTIONS':
        
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Embed the query
        # embedded_query = embeddings.embed_query(user_input,output_dimensionality=384)
        embedded_query=embedding_model.encode(user_input).tolist()
        # Query ChromaDB
        data = data_collection.query(query_embeddings=embedded_query,n_results=7)
        print(data["documents"][0][0])
        data_docs = data["documents"][0]
        text = "".join(data_docs)
        # print(text)
        # Build context from chat history
        context_text = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
        # Rewrite the user query into 2-3 concise variations using the model + parser
        rewrite_prompt = PromptTemplate(
            template="Rewrite the following user query into 2-3 short, clear variations:\n\n{query}",
            input_variables=["query"]
        )
        rewrite_chain = rewrite_prompt | model | parser
        better_query = rewrite_chain.invoke({"query": user_input})
        if not isinstance(better_query, str):
            better_query = str(better_query)
        # Get AI response
        result = chain.invoke({
            'context': context_text,
            'query': better_query,
            'data': text
        })
        
        # Update chat history
        chat_history.append(HumanMessage(content=prompt.format(thing=user_input)))
        chat_history.append(AIMessage(content=result))
        

        # Format history for frontend
        history = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                history.append({'role': 'assistant', 'content': msg.content})
        return jsonify({
            'response': result,
            'history': history
        })
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            history.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            history.append({'role': 'assistant', 'content': msg.content})
    
    return jsonify({'history': history})

@app.route('/clear', methods=['POST', 'OPTIONS'])
def clear_history():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    global chat_history
    chat_history = [SystemMessage(content='You are helpful AI assistant')]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')