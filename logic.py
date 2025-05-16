import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chains import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from prompts.chat_prompt import rag_chat_prompt
from prompts.summary_prompt import synthese_prompt

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def generate_summary(text):
    chain = synthese_prompt | llm
    result = chain.invoke({"text": text})
    return result.content

def setup_qa_chain(vectorstore):
    qa_chain = LLMChain(llm=llm, prompt=rag_chat_prompt)

    stuff_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context"
    )

    return RetrievalQA(
        retriever=vectorstore.as_retriever(),
        combine_documents_chain=stuff_chain,
        return_source_documents=True
    )