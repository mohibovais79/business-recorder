import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import chromadb

import huggingface
import textwrap
import streamlit as st
from langchain.schema import prompt
from langchain.prompts import PromptTemplate


model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation",model=model,tokenizer=tokenizer,max_length=2000,temperature=0,top_p=0.95,repetition_penalty=1.15)
local_llm=HuggingFacePipeline(pipeline=pipe)

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

vectordb = Chroma(persist_directory='db',embedding_function=model_norm)
vectordb.get()
retriever=vectordb.as_retriever(search_kwags={"k":25})

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are a helpful, respectful and honest  assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

instruction = """CONTEXT:/n/n {context}/nQuestion: {question}"""

prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": llama_prompt}


qa_chain=RetrievalQA.from_chain_type(llm=local_llm,chain_type="stuff",retriever=retriever,chain_type_kwargs=chain_type_kwargs,return_source_documents=True)






def wrap_text_preserve_newlines(text,width=110):
    lines=text.split('\n')

    wrapped_lines=[textwrap.fill(line,width=width) for line in lines]
    wrapped_text='\n'.join(wrapped_lines)

    return wrapped_text
def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])



st.title("News Chatbot")

user_input = st.text_input("You:")
if user_input.lower() == "exit":
    st.stop()
response = qa_chain(user_input)
st.text_area("Bot's Response:", response['result'])
