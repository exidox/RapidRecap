from flask import Flask, request, render_template, jsonify
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import TextLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings #OpenAIEMbeddings
from langchain_community.vectorstores import Chroma #it is a type of vector store
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

import bs4
from flask_cors import CORS

from textwrap3 import wrap
import torch

#for conditional generation tasks and input tokenization for t5
from transformers import T5ForConditionalGeneration, T5Tokenizer    

import random
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from flashtext import KeywordProcessor
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer

import string
#import pke
import traceback

from rake_nltk import Rake
from sense2vec import Sense2Vec

from sentence_transformers import SentenceTransformer
from similarity.normalized_levenshtein import NormalizedLevenshtein
from transformers import AutoTokenizer, AutoModel

import keybert
from keybert import KeyBERT
#import yake

from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity





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

@app.route('/mcq', methods=['GET'])
def mcq_req():
    url = request.args.get('url', "")
    loader = WebBaseLoader(web_paths=(url,))
    doc_text = loader.load()
    text = " ".join([doc.page_content for doc in doc_text])
    mcqs = generate_mcqs(text)
    return jsonify(mcqs), 200

def get_summary(text):
    #llm = Ollama(model="llama3")
    llm = Ollama(model = "tinyllama")
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

def generate_mcqs(text):
    summarized_text = get_summary(text)
    
    def rake_extractor(text):
        r = Rake(max_length=3)
        r.extract_keywords_from_text(text)
        lst = np.unique(r.get_ranked_phrases())
        return lst
    
    def get_keywords(original_text, summary_text):
        keywords = (rake_extractor(original_text))
        #print("keywords unsummarized: ", keywords)
        keyword_processor = KeywordProcessor()

        for keyword in keywords:
            keyword_processor.add_keyword(keyword)
        
        keywords_found = keyword_processor.extract_keywords(summary_text)
        keywords_found = list(set(keywords_found))
        #print ("keywords_found in summarized: ",keywords_found)

        important_keywords =[]
        for keyword in keywords:
            if keyword in keywords_found:
                important_keywords.append(keyword)
    
    
        return important_keywords
    imp_keywords = get_keywords(text,summarized_text)

    def filter_keywords_by_pos(keywords):
        filtered_keywords = []
        for keyword in keywords:
            pos_tags = nltk.pos_tag(nltk.word_tokenize(keyword))
            if any(tag in ['NN', 'NNS', 'NNP', 'NNPS'] for _, tag in pos_tags):
                filtered_keywords.append(keyword)
        lemmatizer = WordNetLemmatizer()
        #print(keywords)
        
        final_keywords = []
        for key in filtered_keywords:
            final_keywords.append(lemmatizer.lemmatize(key))
        #print(final_keywords)
        s_list = list(final_keywords)

        for key in s_list:
            if len(key) < 3:
                final_keywords.remove(key)  
        return final_keywords
    
    final_keywords = filter_keywords_by_pos(imp_keywords)

    corpus = brown.sents()
    corpus = [" ".join(sent) for sent in corpus]

    vectorizer = TfidfVectorizer()#ngram_range=(1, 3))
    X = vectorizer.fit_transform(corpus)

    # Get the IDF scores
    idf_scores = vectorizer.idf_

    idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf_scores))

    # Define a threshold for low IDF (common words)
    idf_threshold = np.percentile(idf_scores, 40)

    """ for x in final_keywords:
        if x in idf_dict:
            print(x, idf_dict[x])
    print("thesh",idf_threshold) """
    
    result_keywords = []
    for keyword in final_keywords:
        if keyword in idf_dict:
            if idf_dict[keyword] < float(idf_threshold):
                continue
            else:
                result_keywords.append(keyword)
        else:
                result_keywords.append(keyword)        

    #print("Filtered Keywords with High IDF:", result_keywords)

    result_keywords = np.unique(result_keywords)

    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

    def get_question(context, answer, model, tokenizer):
        text = "context: {} answer: {}".format(context, answer)
        encoding = tokenizer.encode_plus(text, max_length = 1000, pad_to_max_length = False, truncation= True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        out = model.generate(input_ids= input_ids, attention_mask = attention_mask, #early_stopping= True, 
                            num_beams = 3,
                            num_return_sequences =1,
                            no_repeat_ngram_size = 2,
                            max_length = 72)
        
        dec = [tokenizer.decode(ids, skip_special_tokens = True) for ids in out]

        Question = dec[0]
        #Question= Question.strip()
        return Question
    
    #s2v = Sense2Vec().from_disk('S2V/s2v_old')
    def get_answer_and_distractor_embeddings(answer,candidate_distractors):
        model= SentenceTransformer('all-MiniLM-L12-v2')  
        #model = SentenceTransformer('msmarco-distilbert-base-v3')
        answer_embedding = model.encode([answer])
        distractor_embeddings = model.encode(candidate_distractors)
        return answer_embedding,distractor_embeddings
    
    def mmr(answer_embedd, distractor_embedds,words, top_n, diversity):     #Maximal Marginal Relevance
        dis_ans_similarity = cosine_similarity(distractor_embedds, answer_embedd)
        dis_similarity = cosine_similarity(distractor_embedds)

        keywords_idx = [np.argmax(dis_ans_similarity)]
        
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
        #print(candidates_idx)

        for _ in range(top_n - 1):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = dis_ans_similarity[candidates_idx, :]
            target_similarities = np.max(dis_similarity[candidates_idx][:, keywords_idx], axis=1)

            # Calculate MMR
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [(words[idx], round(float(dis_ans_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

    def check_sense_and_distractor(ans):
        s2v = Sense2Vec().from_disk('C:/Users/shubh/OneDrive/Desktop/all/internship/ShorthillsAI/summarizer/RapidRecap/S2V/s2v_old')
        originalword = ans
        word = originalword.lower()
        word = word.replace(" ", "_")

        #print ("word ",word)
        distractors = []

        if s2v.get_best_sense(word):
            sense = s2v.get_best_sense(word)
            #print ("Best sense ",sense)
            most_similar = s2v.most_similar(sense, n=20)
            #print(most_similar)

            for each_word in most_similar:
                append_word = each_word[0].split("|")[0].replace("_", " ")
                if append_word not in distractors and append_word != originalword:
                    distractors.append(append_word)
            distractors.insert(0,originalword)
            sns_fnd= True
            
        else:
            #print("No best sense exits. using other keywords")
            distractors = list(result_keywords)
            distractors.insert(0,originalword)
            sns_fnd= False

        answer_embedd, distractor_embedds = get_answer_and_distractor_embeddings(originalword, distractors)
        final_distractors = mmr(answer_embedd,distractor_embedds,distractors,4, 1)
        filtered_distractors = []
        for dist in final_distractors:
            filtered_distractors.append(dist[0])
        if sns_fnd == False:
            filtered_distractors[2] = "Both of the above"
            filtered_distractors[3] = "None of the above"
        return(filtered_distractors)
    
    nos = 1
    mcq = []
    for ans in result_keywords:
        lst = []
        
        ques = get_question(summarized_text,ans,question_model,question_tokenizer)
        part = ques.split(":")

        first = str(str(part[0]).capitalize() + " " + str(nos) + ":"+str(part[1]))
        lst.append(first)
        nos += 1

        get_distractors = check_sense_and_distractor(ans)
        answer = get_distractors[0]
        
        random.shuffle(get_distractors)
        lst.append("a.) " + get_distractors[0].capitalize())
        lst.append("b.) " + get_distractors[1].capitalize())
        lst.append("c.) " + get_distractors[2].capitalize())
        lst.append("d.) " + get_distractors[3].capitalize())
        lst.append("----------")
        lst.append("Answer :"+ answer.capitalize())
        lst.append(" ")

        mcq.append(lst)

    return mcq    
    
@app.route('/answer_ques', methods=['GET'])
def answer_ques():
    print("query received")
    query = request.args.get('query')
    url = request.args.get('url')

    if not query:
        return "No query provided", 400
    
    loader = WebBaseLoader(web_paths = (url,))
    text_documents = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 20)
    documents = text_splitter.split_documents(text_documents)

    print("starting embeddings")

    db= Chroma.from_documents(documents, OllamaEmbeddings())
    print("embeddings done")
    
    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    <context>
    {context}
    </context>
    Question: {input}""")       #context-> all documents in the vector store, input->question asked


    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = db.as_retriever()

    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("retrieving")
    response = retrieval_chain.invoke({"input": query})

    print("returning answer")
    return str(response["answer"]), 200



if __name__ == "__main__":
    app.run(debug="True")
