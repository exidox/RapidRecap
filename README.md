# RapidRecap
1. Introduction to Rapid-Recap
Rapid-Recap is a Chrome extension for effortlessly digesting large amounts of information from the web. Whether you’re a student, a professional, or an avid reader, Rapid-Recap is designed to enhance your online learning experience by providing concise summaries of lengthy web pages, generating multiple-choice questions for practice, and allowing you to ask questions about the text you’re reading. It simplifies your study sessions, improves comprehension, and makes the most out of your online learning. 

1.1 Key Features:

a.) Summarize Large Web Pages: Quickly get the gist of extensive articles, research papers, and other lengthy content with our advanced summarization feature. Save time and grasp the key points without wading through endless paragraphs.

b.) Generate MCQs to Test Yourself: Enhance your learning and retention by generating multiple-choice questions from the text. Perfect for students preparing for exams or anyone looking to reinforce their understanding of the material.

c.) Interactive Q&A: Have questions about what you’re reading? Simply ask, and Rapid-Recap will provide clear and accurate answers, helping you gain deeper insights and clarify any doubts.

2. Methodology:
Rapid-Recap is built using Generative AI and NLP. The procedure is as follows:
a.) Generating Summary: 
1.) TinyLlama is used as the LLM here. It is chosen due to its smaller size without compromising the efficiency. Llama3 could be an alternative for better results, however it is much more resource intensive.
2.) Diving the entire texts using Recursive Text Splitter to create chunks.
3.) Making a template to write the prompt i.e  “Write a summary focusing on the key headings and important topics:   Text : `{text}` ”
4.) Loading summarize chain and using “map reduce” for multiple chunks of text.

b.) Generating MCQs:
1. )First create a summary.
2.) Extract important keywords from the entire text using Rake.
3.) FInding common words between the keywords extracted from the main text and the summarized text.
4.) Filter words from POS, retaining only noun phrases.
5.) Further filter the words by evaluating their IDF values in the “brown corpus”. Only keywords with high IDF values were selected.
6.) Using a fine tuned Question Generation Model. Here, “ramsrigouthamg/t5_squad_v1” model is used which was used by fine tuning T5 LLM on Squad Dataset. The keywords extracted acted as the answers through which the questions were generated.
7.) To generate the other options in the MCQ, distractors are generated. Sense2Vec is used to find similar semantic words. Then using Maximal Marginal Relevance, extract similarities within candidates and between candidates and selected keywords/phrases.
   
c.) Interactive Q&A:
1.) Retrieval Augmented Generation is used to answer the queries based on the context (i.e contents of the webpage).
2.) Splitting the entire webpage into document chunks.
3.) Creating embeddings into the Chroma Database.
4.) Using Llama3 and a prompt template "Answer the following question based only on the provided context. <context>  {context} </context> Question: {input}" the create the document chain.
5.) Retrieving the answer of the query using the chain.




