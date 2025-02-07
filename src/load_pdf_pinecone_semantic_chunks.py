import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
import asyncio
import re
import time
import pickle
import random
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone
from pinecone import ServerlessSpec



load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
data_path = os.getenv('PDF_PATH')





## READING THE PDF FILES

async def load_pages(pdf_folder):

    files = os.listdir(pdf_folder)

    pages = []

    for file in files:
        
        file_path = str(os.path.join(pdf_folder, file))

        loader = PyPDFLoader(file_path)
        
        async for page in loader.alazy_load():
            pages.append(page)


    return pages


data = asyncio.run(load_pages(data_path))

print(type(data))


# FUNCTION TO CLEAN THE TEXT

def clean_docs(docs):
    clean_docs = []

    for doc in docs:
        doc.page_content = re.sub(r"\n", r" ", doc.page_content)
        doc.page_content = re.sub(r"[' ']+", r" ", doc.page_content)
        clean_docs.append(doc)

    return clean_docs


cleaned_data = clean_docs(data)
print(">> Finished text cleaning\n")



# Initializing Semantic Chunker

text_splitter = SemanticChunker(HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-miniLM-L6-v2",
            model_kwargs = {'device':'cpu'}),
            
        breakpoint_threshold_type = 'gradient',
        )


split_data = text_splitter.split_documents(cleaned_data)




# SAVING THE CHUNKED TEXT INTO A PICKLE FILE

with open(r"/aviation_text_semchunked.pkl", 'wb') as f:
     pickle.dump(split_data, f)


print(">> Finished creating pickle file\n")





# Embedder class

class DocumentEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs = {'device':'cpu'}
            )


    def embed_documents(self, docs):
        texts = [doc.page_content for doc in docs]
        return self.model.embed_documents(texts)

    def embed_query(self, doc):
        return self.model.embed_query(doc)



embedder = DocumentEmbedder("sentence-transformers/all-MiniLM-l6-v2")


doc_embeddings = embedder.embed_documents(split_data)
print(">> Finished embedding docs\n")

doc_metadata = [doc.metadata for doc in split_data]
metadata = []
for i,dict in enumerate(doc_metadata):
     dict.update({'raw_text_index':str(i)})
     metadata.append(dict)
     



#### PINECONE VECTOR DB

pc = Pinecone(api_key=pinecone_api_key)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)


embed_dim = 384

index_name = 'aviation-info-semantic-chunking'



# Create Pinecone Index

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
        index_name,
        dimension=embed_dim,  
        metric='euclidean',
        spec=spec
    )
# wait a moment for the index to be fully initialized
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)



index = pc.Index(index_name)


## UPSERTING with batch size 1000

print(">> Starting Pinecone upsert\n")

batch_size = 1000
data_vectors = []

for i, (embedding, meta) in tqdm(enumerate(zip(doc_embeddings, metadata))):
            


            data_vectors.append({
                "id": str(i),
                "values": embedding,
                "metadata": meta
            })

            if len(data_vectors) >= batch_size:
                index.upsert(vectors=data_vectors)
                data_vectors = []  # Reset the batch


if data_vectors:
    index.upsert(vectors=data_vectors)

print(f">>> UPSERT COMPLETED!")