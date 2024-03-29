from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate

import re
import os
import dotenv
dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
TOKEN_MAX = os.getenv("TOKEN_MAX")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")

# Function to remove URLs from text
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

text_file_path = "./conversation.txt"
with open(text_file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    text_without_urls = remove_urls(text)

# Load the text file
loader = TextLoader(text_file_path, encoding='utf-8')
docs = loader.load()

# Use gpt-4 model which has a token limit of 8192
llm = ChatOpenAI(temperature=0, model='gpt-4', api_key=API_KEY)

# Create LLMChain which is for individual summary
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Create LLMCHain which is for reducing summary
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Create a final chain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Create a chain to Combine and iteratively reduce the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=TOKEN_MAX,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

# Creat a text splitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

summary = map_reduce_chain.invoke(split_docs)
print(summary['output_text'])