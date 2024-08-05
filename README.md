# RapidRecap

## 1. Introduction to Rapid-Recap
Rapid-Recap is a Chrome extension for effortlessly digesting large amounts of information from the web. Whether you’re a student, a professional, or an avid reader, Rapid-Recap is designed to enhance your online learning experience by providing concise summaries of lengthy web pages, generating multiple-choice questions for practice, and allowing you to ask questions about the text you’re reading. It simplifies your study sessions, improves comprehension, and makes the most out of your online learning. 

1.1 Key Features:
a.) Summarize Large Web Pages: Quickly get the gist of extensive articles, research papers, and other lengthy content with our advanced summarization feature. Save time and grasp the key points without                wading through endless paragraphs.

b.) Generate MCQs to Test Yourself: Enhance your learning and retention by generating multiple-choice questions from the text. Perfect for students preparing for exams or anyone looking to reinforce their             understanding of the material.

c.) Interactive Q&A: Have questions about what you’re reading? Simply ask, and Rapid-Recap will provide clear and accurate answers, helping you gain deeper insights and clarify any doubts.

## 2. Methodolgy
Rapid-Recap is built using Generative AI and NLP and then deployed using Flask with a user interface developed through HTML, CSS and JS. The procedure is as follows:
### Generating Summary: 
TinyLlama is used as the LLM here. It is chosen due to its smaller size without compromising the efficiency. Llama3 could be an alternative for better results, however it is much more resource intensive.
Diving the entire texts using Recursive Text Splitter to create chunks.
Making a template to write the prompt i.e  “Write a summary focusing on the key headings and important topics:   Text : `{text}` ”
Loading summarize chain and using “map reduce” for multiple chunks of text.

### Generating MCQs:
First create a summary.
Extract important keywords from the entire text using Rake.
FInding common words between the keywords extracted from the main text and the summarized text.
Filter words from POS, retaining only noun phrases.
Further filter the words by evaluating their IDF values in the “brown corpus”. Only keywords with high IDF values were selected.
Using a fine tuned Question Generation Model. Here, “ramsrigouthamg/t5_squad_v1” model is used which was used by fine tuning T5 LLM on Squad Dataset. The keywords extracted acted as the answers through which the questions were generated.
To generate the other options in the MCQ, distractors are generated. Sense2Vec is used to find similar semantic words. Then using Maximal Marginal Relevance, extract similarities within candidates and between candidates and selected keywords/phrases.
### Interactive Q&A:
Retrieval Augmented Generation is used to answer the queries based on the context (i.e contents of the webpage).
Splitting the entire webpage into document chunks.
Creating embeddings into the Chroma Database.
Using Llama3 and a prompt template "Answer the following question based only on the provided context. <context>  {context} </context> Question: {input}" the create the document chain.
Retrieving the answer of the query using the chain.

### Important Libraries used:
        Langchain
        
        Transformers

        NLTK

        Numpy

        Sklearn

        Torch

        Flask

## 3. Extension Components:
### HTML file:
The popup.html file acts as the extension window. 
You get a brief description of the application as well as 3 buttons to generate summary, MCQs and clear the output.
An input textbox allows users to ask relevant questions from the current webpage/context.
### CSS file:
The popup.css file provides a clean and minimalistic design for the application.
### JavaScript file:
The popup.js file encompasses the handling of the DOM and manipulation.
It handles all the HTTP Requests and Responses.
It makes our extension dynamic.

## 4. Installation and Setup:
To use this tool, follow the steps below:
1. Clone this repository to your local machine and download it:
        https://github.com/exidox/RapidRecap
   
3. Install the required dependencies from requirements.txt or install the following using pip:
flask

        flask-cors 

        langchain 

        langchain_community 

        beautifulsoup4 

        textwrap3 

        torch 

        transformers 

        numpy 

        nltk 

        flashtext

        scikit-learn 

        rake-nltk 

        sense2vec==1.0.3

        sentencepiece==0.1.95

        sentence-transformers

        keybert 

        chromadb

5. Download the sense2vec pretrained vectors archive and pass the extracted directory to Sense2Vec.from_disk.
        https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz

6. Extract the file.

7. Change the file path to access the vectors in app.py file within the “check_sense_and_distractor” function.

## 5. Usage:
To use Rapid-Recap, follow the steps below:
1. Run the main script:
        app.py
2. Open the chrome browser and search “chrome://extensions/”.
3. Click on “Load unpacked” and upload the Rapid-Recap extension folder.
4. Now open the webpage you want to use the extension on.
5. Click on the extensions button and open Rapid-Recap.
6. You can now use Rapid-Recap for your intended purpose, such as studying, creating practice tests, or integrating them into an educational platform.


