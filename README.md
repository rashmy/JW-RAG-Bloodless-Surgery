import streamlit as st 
from langchain_community.document_loaders import DirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_community.llms import Ollama 
from langchain_core.prompts import PromptTemplate 
# ------------------------------- 
# 1. Load & Prepare Vector Store 
# ------------------------------- 

DATA_FOLDER = "D:/JW_RAG_Project/data_raw" 

@st.cache_resource 
def load_vectorstore(): 
   loader = DirectoryLoader(DATA_FOLDER, glob="*.txt") 
   documents = loader.load() 

   text_splitter = RecursiveCharacterTextSplitter( 
   chunk_size=256, 
   chunk_overlap=50 
   ) 
   chunks = text_splitter.split_documents(documents) 
   
   embeddings = HuggingFaceEmbeddings( 
     model_name="sentence-transformers/all-MiniLM-L6-v2" 
   ) 

   vectorstore = FAISS.from_documents(chunks, embeddings) 
   return vectorstore.as_retriever(k=5) 

retriever = load_vectorstore() 
# ------------------------------- 
# 2. Initialize LLM & Prompt 
# ------------------------------- 

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
# 3. Streamlit UI 
# ------------------------------- 

st.set_page_config(page_title="JW RAG Chatbot Evaluation", layout="centered") 
st.title("📘 JW Bloodless Medicine — RAG Chatbot Evaluation") 

st.write("Test your chatbot with multiple questions and see retrieval + answer accuracy.") 

if "results" not in st.session_state: 
  st.session_state.results = [] 

# Input box for multiple questions 
question = st.text_input("Enter your question:", "") 

expected_answer = st.text_input("Expected answer (for evaluation, optional):", "") 

if st.button("Test Question"): 
  if question: 
  # Retrieve chunks 
  docs = retriever.invoke(question) 
  context = "\n\n".join([d.page_content for d in docs]) 
  retrieved_ids = [str(i) for i in range(len(docs))] 

  # Generate answer 
  answer = chain.invoke({"context": context, "question": question}) 

  # Simple scoring: check if expected answer is in generated answer answer_correct = 0 
  if expected_answer.strip() != "": 
     answer_correct = int(expected_answer.lower() in answer.lower()) 

  # Store results 
  st.session_state.results.append({ 
    "question": question, 
    "expected_answer": expected_answer, 
    "generated_answer": answer, 
    "retrieved_chunks": len(docs), 
    "answer_correct": answer_correct }) 

# Display results 
if st.session_state.results: 
   st.markdown("### Test Results") 
   for i, r in enumerate(st.session_state.results): 
     st.markdown(f"**Q{i+1}:** {r['question']}") 
     if r['expected_answer']: 
       st.markdown(f"**Expected:** {r['expected_answer']}") 
     st.markdown(f"**Generated:** {r['generated_answer']}") 
     st.markdown(f"**Chunks Retrieved:** {r['retrieved_chunks']}") 
     if r['expected_answer']: 
       st.markdown(f"**Answer Correct:** {r['answer_correct']}") 
     st.markdown("---") 



