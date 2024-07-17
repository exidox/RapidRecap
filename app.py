from flask import Flask, request, render_template
#from langchain import PromptTemplate
#from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import bs4
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/summary', methods=['GET'])
def url_req():
    url = request.args.get('url', "")
    loader = WebBaseLoader(web_paths = (url,))
    doc_text = loader.load()
    text = " ".join([doc.page_content for doc in doc_text])
    summary = get_summary(text)
    return summary, 200

def get_summary(text):
    llm = llm = Ollama(model="llama3")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=20)
    chunks = text_splitter.create_documents([text])
    generic_template='''
    Write a summary of the following text in 500 words keeping in mind to focus on the key headings and important topics:
    Text : `{text}`
    '''
    prompt=PromptTemplate(
        input_variables=['text'],
        template=generic_template
    )
    chain= load_summarize_chain(llm, chain_type="map_reduce", verbose = True, combine_prompt=prompt)
    summary = chain.run(chunks)
    return summary


if __name__ == "__main__":
    app.run(debug="True")
