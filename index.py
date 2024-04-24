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
TOKEN_MAX = int(os.getenv("TOKEN_MAX"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))

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
map_template = """
{docs}
The documents provided contain extensive discussions from a cryptocurrency-focused Telegram group chat. Your task is to identify key themes and summarize the content with the following structure.
Explicitly ignore all low-information content such as greetings, casual banter, or repetitive trade alerts that do not contribute substantial information. The summary should be efficient, prioritizing depth and relevance to provide a clear snapshot of the group's discussions and deliver practical knowledge and insights to the reader.
Core Discussion Themes:
- Identify the most prevalent and impactful topics discussed, especially those relevant to current market trends, technological advancements, or regulatory changes in cryptocurrency.
- Exclude routine trade alerts and transient price discussion unless they encapsulate broader market sentiment or trends.
Strategic Insights:
- Highlight strategic discussions that offer depth, such as risk management practices, trading strategies, or investment philosophies.
- Summarize perspectives on market predictions, analysis of coin potential, or debates on blockchain technologies.
Educational Content:
- Extract and summarize educational exchanges where users share insights, explain concepts, or debunk common misconceptions (e.g., explanations of "FUD", investment tactics, or technology applications).
Key Contributors and Opinions:
- Note influential contributors and summarize their most insightful comments, especially those providing unique insights or expert analysis.
- Provide context on their influence or expertise if mentioned or evident from discussions.
Community Sentiment and Reaction:
- Gauge and report the overall sentiment of the community regarding major events or announcements (e.g., regulatory news, major hacks, or significant market moves).
- Note shifts in sentiment and their potential triggers.
Actionable Takeaways:
- List actionable advice or consensus views that could benefit someone looking to understand or navigate the cryptocurrency market.
- Include any commonly recommended tools, resources, or strategies discussed among members.
"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Create LLMChain which is for reducing summary
reduce_template = """
{docs}
Given the individual summaries generated, consolidate them into a final, comprehensive summary while preserving the original structure. 
List the result with the following items.
Core Discussion Themes:
Strategic Insights:
Educational Content:
Key Contributors and Opinions:
Community Sentiment and Reaction:
Actionable Takeaways:
"""
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