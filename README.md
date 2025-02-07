# AviationRAG


This project aims to build a RAG-based LLM assistant for airline pilots and members of technical crew.


The completed database will consist of PDF documents of Aircraft Flight Documentation (AFM), technical manuals, weather guides, navigation guides etc. collected from official/verified sources like Federal Aviation Administration and Flight Manuals Online.

Currently, the dataset consists of the following PDF files:
   * Aviation Emissions and Air Quality Handbook Version 4
   * Information for Pilots Considering Laser Eye Surgery.
   * Airbus A320 Flight Crew Manual
   * Aeronautical Information Manual 2024


This repository demosntrates the RAG pipeline with GPT 3.5-Turbo, along with two modifications to the RAG pipeline:

* Semantic Chunking - Involves creating semantically separated chunks from the text documents instead of the regular method of character based splitting


* Contextual Compression - Compressing the retrieved documents using LLM to create a concise context for LLM response generation



### Steps to run code:


1. Download the PDF files from the [Drive](https://drive.google.com/drive/folders/1zKwDvTcqjksVpf6BcFKIgpq21fbXR7cU?usp=sharing) and store on local
    

2. Run requirements.txt:
   
   ```!pip install requirements.txt```


3. Update environment variables:
   * Pinecone API key
   * OpenAI API key
   * PDF folder path


4. Create the Pinecone vector stores by running ```load_pdf_pinecone.py ``` and ```load_pdf_pinecone_semantic_chunks.py ```
   * Edit the pickle file save paths within the code


5. Run ```openai_chat.py``` to get responses from Chat GPT-3.5


6. Run ```retriever.py``` to get LLM responses with regular RAG and RAG with semantic chunking applied


7. Run ```retriever_with_compression.py``` to get LLM responses with RAG along with contextual compression
