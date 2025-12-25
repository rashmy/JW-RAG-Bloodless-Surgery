#source code

from langchain_community.document_loaders import DirectoryLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
import streamlit as st 

# ------------------------------- 
# 1. Load & Prepare Vector Store 
# ------------------------------- 

DATA_FOLDER = "D:/JW_RAG_Project/data_raw" 

@st.cache_resource 
def load_vectorstore(): 
   # Load documents from folder
   loader = DirectoryLoader(DATA_FOLDER, glob="*.txt") 
   documents = loader.load() 

   # Split text into manageable chunks
   text_splitter = RecursiveCharacterTextSplitter( chunk_size=256, chunk_overlap=50) 
   chunks = text_splitter.split_documents(documents) 
   
   # Generate embeddings
   embeddings = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2") 

   # Store in FAISS vector database
   vectorstore = FAISS.from_documents(chunks, embeddings) 
   return vectorstore.as_retriever(k=5) 

retriever = load_vectorstore() 

# ------------------------------- 
# 2. Initialize LLM & Prompt 
# ------------------------------- 
from langchain_community.llms import Ollama 
from langchain_core.prompts import PromptTemplate 

llm = Ollama( 
  model="jw", 
  temperature=0.1 
) 

prompt = PromptTemplate( 
  template=""" 
You are an AI assistant. Answer the question using ONLY the context below. 
Do NOT use your own knowledge. 

Context: 
{context} 

Question: 
{question} 

If not found in the context, say "I don't know". 
""", 
  input_variables=["context", "question"] 
) 

chain = prompt | llm 

# ------------------------------- 
# -------------------------------
# 3. Streamlit UI 
# -------------------------------

st.set_page_config(page_title="JW RAG Chatbot", layout="centered", )

st.title("📘 JW Bloodless Medicine — RAG Chatbot")
st.write("Ask any question related to Bloodless Medicine. The system will answer using information retrieved from the dataset.")

# Input box for question
question = st.text_input("Enter your question:", "")

if st.button("Get Answer"):
    if question:

        # Retrieve relevant chunks
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        # Generate answer from RAG pipeline
        answer = chain.invoke({"context": context, "question": question})

        # Display results
        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved Context (Top 5 Chunks)")
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(d.page_content)
            st.markdown("---")


