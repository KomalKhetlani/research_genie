# Research Genie

** Your AI Assistant for NLP & Large Langauge Models. **

Research Genie is a Retrieval Augmented Generation (RAG) based chatbot that answers NLP and LLM related queries using **LLAMA3 , OLLAMA and CHROMADB**

---

Features

**Conversational Chatbot** - Engages in interactive Q&A with memory 
**Retrieval-Augmented Generation (RAG)** - Uses ChromaDb for document retrieval 
**Powered by LLAMA3 & OLLAMA** - generates high quality NLP responses
**Fast & Efficient** - Avoids retrival for chitchat queries
**User-Friendly UI** - Built with Streamlit

## Install Dependencies 

pip install -r requirements.txt

## Run the application

streamlit run app.py

## Using Ollama for LLAMA3

Ensure Ollama is installed and running. If not, install it from:
https://ollama.ai/

Then download LLama3 using

ollama pull llama3

## How it works

User enters a query in the chatbot UI
chitchat queries are answered directly ising LLAMA3
For complex and NLP related queries. Relevant NLP documents are retrieved from ChromaDB
LLAMA3 generates a cohesive answer for the retrieved content
The conversation history is maintained , allowing a seamless chat experience.

## License
This project is open-source and available under MIT License.

## Contact
Komal Khetlani 
Email: komalkhetlani2525@gmail.com
Linkedin : https://www.linkedin.com/in/komal-khetlani/





