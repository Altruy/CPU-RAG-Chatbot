from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# from llm_ import LlamaLLM
import streamlit as st
from utils_chromadb import ChromaVDB
from langchain_community.llms import LlamaCpp

import time

# print(llm.invoke("Tell me a joke"))
load_dotenv()


        
model = "../model-cache/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
mis = """
[INST] Use General knowledge and mainly this context: {context} 
to answer the query: {query} [/INST]
"""
llama_3_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Your task is to answer the user's query
with as much detail as possible using the context provided.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
context: {context}
query: {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

ollama_3_template = """use the following your own knowledge and mainly this context: "{context}"
to answer the user query: "{query}" """
phi_3 = """<|system|>
You are a helpful assistant Question ANswer Bot. Your task is to answer the user's Questions
with as much detail as possible using the context from Text files provided. <|end|>
<|user|>
Context: {context}

Question: {query}
Answer: <|end|>
<|assistant|>"""

prompt_ = PromptTemplate.from_template(llama_3_template)


st.title("QA Chatbot")
st.caption("A chatbot for the document Management.txt")



@st.cache_resource
def load_model ():
    llm = LlamaCpp(
        model_path="../model-cache/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=8192,
        n_ctx=8192,
        top_p=1,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    # llm = Ollama(model="llama3")
    vdb = ChromaVDB(persistant=False)
    vdb.add_to_db('change-management.pdf')
    return llm,vdb


with st.spinner('Model Loading...'):
    llm, vdb = load_model()


@st.experimental_fragment
def stre():
    con = st.container(height=500,border=True)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        con.chat_message("assistant").write("How can I help you?")

    for msg in st.session_state.messages:
        con.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})
            con.chat_message("user").write(prompt)
            results = vdb.retrieve_data(prompt,top_k=4)

            if len(results) > 0:
                context = '\n\n'.join([doc[0] for doc in results['documents']])
            else: context = "\n"
            
            response = llm.invoke(prompt_.format(context=context,query=prompt))
            msg = response
            st.session_state.messages.append({"role": "assistant", "content": msg})
            con.chat_message("assistant").write(msg)
        except Exception as e:
            st.error(str(e))

stre()