import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()



# Initialize GPT 3.5 chat model

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Defime prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers user queries.",
        ),
        ("human", "{input}"),
    ]
)


chain = prompt | llm


# Function to run the chain

def run_chain(query):
    return chain.invoke(
    {
        "input": query,
    }
).content




if __name__ == "__main__":

    with open (r"../sample_questions.txt", 'rb') as f:
        lines = f.readlines()

    lines = [i.strip() for i in lines]


    for line in lines:
        print(f"\n\n>> GPT Response: {run_chain(line)}")
