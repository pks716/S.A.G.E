import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question,db):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    
    new_db = FAISS.load_local(db, embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("S.A.G.E")
    st.header("S.A.G.E - Student's Academic Guide Engine :books:")

    vectorstore_path = "vector_store/"
    subfolders = [f.name for f in os.scandir(vectorstore_path) if f.is_dir()]
    subfolders_string = ', '.join(subfolders)
    option = st.selectbox(
    'Select a subject first:',
    (list(subfolders)))  

    user_question = st.text_input("Ask a Question from the selected Subject")

    

    if user_question:
        user_input(user_question,vectorstore_path+"/"+option)



if __name__ == "__main__":
    main()
