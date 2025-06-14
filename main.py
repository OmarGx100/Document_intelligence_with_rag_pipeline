import streamlit as st
from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio
import time
from typing import List
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain_community.vectorstores 
from langchain_ollama.embeddings import OllamaEmbeddings
import os
from langchain_postgres import PGVector
import langchain_ollama.embeddings
from vector_database_faiss import FaissVectorDB
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from UserInent import RAGPipeline

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://postgres:zatona123@localhost:5432/postgres"  # Uses psycopg3!
collection_name = "langchain_vector_db_1"




#Load environment variable
env_path = os.path.join(".env")
load_dotenv(env_path)

# loading environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
AZURE_DI_API_KEY = os.getenv("AZURE_DI_API_KEY")

# Instintiate OpenAI object
# client = AsyncOpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Defining Model Name that is going to be used
user_history = []


def generate_embeddings():
    embeddings_model = OllamaEmbeddings(model="paraphrase-multilingual:278m")
    return embeddings_model




# def create_vector_database_pgvector(raw_texts):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 1000, chunk_overlap = 100
#     )

#     vector_store = PGVector(
#     embeddings=generate_embeddings(),
#     collection_name=collection_name,
#     connection=connection,
#     use_jsonb=True,
#     embedding_length=768)


#     texts = text_splitter.split_text(raw_texts)


#     vector_ids = vector_store.add_texts(texts)

#     return vector_store, vector_ids



def parse_file_data_with_document_intellegent(document,
                                              output_content_format = 'text'):
    features = []


    document_intellegence_client = DocumentIntelligenceClient(endpoint="https://omaropenaitest.cognitiveservices.azure.com/",
                                                                            credential= AzureKeyCredential(key = AZURE_DI_API_KEY)) 
    poller = document_intellegence_client.begin_analyze_document("prebuilt-layout",
                                                                 document,
                                                                 content_type="application/octet-stream",
                                                                 output_content_format = output_content_format,
                                                                 features=features)
    results = poller.result()

    if results:
        return results
    else:
        return None


# def semantic_chunking(json_DI_document):
#     paragraphs = json_DI_document['paragraphs']
#     tables = json_DI_document['tables']
#     chunks = []

#     for table in tables:
#         chunks.append(str(table))
#     for parag in paragraphs:
#         chunks.append(str(parag))

#     return chunks



def pdf_reader(pdf_document):
    reader = PdfReader(pdf_document)
    raw_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def chat_with_gpt(model = MODEL_NAME, history = [{"role" : "user", "content" : "this is a test"}]):

    try:
        stream = client.chat.completions.create(
            model = model,
            messages= history,
            temperature=TEMPERATURE,
            stream = True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"An Exception Occured While trying to reach openAI server, Exception Details : {e}")
        return
    

def init_history():
    if "user_history" not in st.session_state:
        st.session_state.user_history  = []
    else:
        k = len(st.session_state.user_history )
        for msg in st.session_state.user_history :
            st.chat_message(msg['role']).write(msg['content'])   

    return

def update_history(role : str, response_text : str, history : List):
    history.append({"role": role, "content": response_text})
    return 




def main():

    st.title("Ask the AI")
    init_history()
    user_input =  st.chat_input("Type your message here...")

    if user_input:
        update_history("user", user_input, st.session_state.user_history)
        with st.chat_message("user"):
            st.markdown(user_input)
        response_text = ""
        response_placeholder = st.empty()
        with st.spinner("AI is generating a response..."):
            for response_part in chat_with_gpt(history=  st.session_state.user_history):
                response_text += response_part
                response_placeholder.chat_message("assistant").write(response_text)
            update_history("assistant", response_text, st.session_state.user_history)



if __name__ == "__main__":
#     main()

    # file_path = "/home/omar/Desktop/R&D/AI/NLP/openai/Omar_Abdel_Rahman.pdf"
    # with open(file_path, 'rb') as pdf:
    #     document_intellengence_data = parse_file_data_with_document_intellegent(pdf, 'text')
    # #     raw_texts = pdf_reader(pdf)
    #     pdf.close()


    

    # semantic_chunks = semantic_chunking(document_intellengence_data)
    db_name = "DOCUMENT_INTELLEGENCE_VECT_DB"
    db_path = "./databases/"
    chunk_size = 1024
    chunk_overlap = 50
    embedding_model_name = "paraphrase-multilingual:278m"


    vector_db = FaissVectorDB(db_name = db_name,
                            db_path = db_path,
                            chunk_size = chunk_size,
                            chunk_overlap = chunk_overlap,
                            embedding_model_name = embedding_model_name)
    
    SUCESS = vector_db.load_vector_db()
    # print(vector_db.get_database_info())
    if not SUCESS:
        # SUCESS = vector_db.create_from_document_intelligence_output(document_intellengence_data)
        vector_db.save_vector_db()
        if SUCESS:
            print("Successfully built Vector DB")
        else:
            print("Could not Build Vector DB")
    else:
        print(f"Successfully loaded database : {vector_db.db_name}")

    
    rag = RAGPipeline(vector_db=vector_db, llm_client= client)

    # query = "who is the owner's name of the provied cv, and where does he/she lives?"
    # query = "what tech feild would this candidate be good at ?"
    query = "tell me about the candidate knowledge about computer vision, and rate him for a senior computer vision engineer position !"

    results = rag.query(user_query=query, k = 10, explain_search=True)

    print("FFFFFFFFFFFFFFFFFFF", results['answer'])




    # # vector_db, vector_ids = create_vector_database_pgvector(raw_texts)
    # query = "مزايا استخدام الطوب"

    # documents = vector_db.similarity_search_with_score(query,  k= 4)


    # for idx, (document, score) in enumerate(documents):
    #     print(f"this is document {idx} has a score of {score}: {document}")