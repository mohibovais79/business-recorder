import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
import chromadb
import pickle
import textwrap
import streamlit as st


model_name='TheBloke/wizardLM-7B-HF'
tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")
model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",load_in_8bit=True,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True )
pipe = pipeline("text-generation",model=model,tokenizer=tokenizer,max_length=2000,temperature=0,top_p=0.95,repetition_penalty=1.15)
local_llm=HuggingFacePipeline(pipeline=pipe)

with open('data.pkl','rb') as f:
    data=pickle.load(f)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts=text_splitter.split_documents(data)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
persist_directory='db'
embedding=instructor_embeddings
vectordb=Chroma.from_documents(documents=texts,embedding=embedding,persist_directory=persist_directory)
retriever=vectordb.as_retriever(search_kwags={"k":3})
qa_chain=RetrievalQA.from_chain_type(llm=local_llm,chain_type="stuff",retriever=retriever,return_source_documents=True)

def wrap_text_preserve_newlines(text,width=110):
    lines=text.split('\n')

    wrapped_lines=[textwrap.fill(line,width=width) for line in lines]
    wrapped_text='\n'.join(wrapped_lines)

    return wrapped_text
def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources: ')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])



st.title("News Chatbot")

user_input = st.text_input("You:")
if user_input.lower() == "exit":
    st.stop()
response = qa_chain(user_input)
process_llm(response)
st.text_area("Bot's Response:", response)
