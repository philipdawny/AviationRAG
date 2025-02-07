import warnings
warnings.filterwarnings("ignore")


import os
from dotenv import load_dotenv
import pickle
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec




load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')


# Initializing Pinecone Vector Store


cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)



index_name = 'aviation-info'




def initialize_vector_store(api_key: str, 
                          index_name: str,
                          model_name: str = "sentence-transformers/all-MiniLM-l6-v2"):

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize vector store from existing index
    vector_store = PineconeVectorStore(
        # pinecone_api=pc,
        pc.Index(index_name),
        # index_name=index_name,
        embedding=embeddings,
        text_key = "raw_text_index"
    )
    
    

    return vector_store


vector_store = initialize_vector_store(
        api_key=pinecone_api_key,
        index_name=index_name
    )



# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", seed=0)


# Opening split data
with open(r"/aviation_text_chunked.pkl", 'rb') as f1:
    splits = pickle.load(f1)



# Retriever class

class CustomRetriever(BaseRetriever):
    vectorstore: PineconeVectorStore
    splits: List

    def _get_relevant_documents(self, query):
        # Perform vector search
        print(f"\n\n>> Query: {query}")
        docs = self.vectorstore.similarity_search(query, k=3)

        print(f"\nDOCS: {len(docs), docs}")

        # Fetch raw text
        outputs=[]
        for doc in docs:
            raw_text = self.splits[int(doc.id)]
            outputs.append(raw_text)
        print(f"\n\n>>> Number of retrieved docs: {len(outputs)}\n>>> Retrieved Docs: {outputs}\n\n")
        return outputs

retriever = CustomRetriever(vectorstore=vector_store, splits=splits)



# Contextual Compression Block

compressor_llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(compressor_llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)




# Defining custom chat prompt template

# Create the RAG prompt template
template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant for question-answering tasks. Use the following context to answer the user's question.
If you don't know the answer based on the context, just say you don't know.
Context: {context}"""),
    ("human", "{question}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Function to run RAG Chain

def run_rag_chain_with_compression(prompt, retriever = retriever):

    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(prompt)

    return response



if __name__ == "__main__":

    with open (r"../sample_questions.txt", 'rb') as f:
        lines = f.readlines()

    lines = [i.strip() for i in lines]


    for line in lines:

        print(f"\n\n>> USER QUERY: {line}")

        print(f"\n\n>> RAG LLM response: {run_rag_chain_with_compression(line)}")
