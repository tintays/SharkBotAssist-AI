#This project was developed for the Assignment 2: Developing an FAQ Chatbot Using LangChain and LLM APIs by Josue Lopez Guevara for the course CS311 - Artificial Intelligence

#Libraries we will be using
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 2. Set API Key
# You need your own key to connect with the Gemini model
os.environ["GOOGLE_API_KEY"] = "Your key here :)"

print("API and environment ready.")

# 1. We initialize local embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the Shark Manual
# Ensure "RV2100_Series_IB_REV_Mv4_251027_HR.pdf" is in the same folder
loader = PyPDFLoader("RV2100_Series_IB_REV_Mv4_251027_HR.pdf")
documents = loader.load()

# 3. Split and Index
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
vector_db = FAISS.from_documents(chunks, embeddings)

print(f"Block 2: Success! {len(chunks)} manual sections indexed.")

# 1. Target the Experimental endpoint from your dashboard
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
)

# 2. RAG Logic
template = """Answer the question based ONLY on the provided manual context:
{context}

Question: {question}

Helpful Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Build the Chain
rag_chain = (
    {"context": vector_db.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Execute
print("--- SHARK ROBOT SUPPORT ---")
try:
    query = input("What do you need me to help you with?\nAsk your question here: ")
    response = rag_chain.invoke(query)
    print(f"\nResponse: {response}")
except Exception as e:
    print(f"\n API Error: {e}")