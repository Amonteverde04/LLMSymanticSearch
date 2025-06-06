from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import chain
from typing import List
import asyncio
import getpass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Load document.
file_path = "example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split document.
# Ensures that the meanings of relevant portions of the document are not "washed out" by surrounding text.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Embed split unstructured pdf text as a vector so we can store it.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

# Store in memory.
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

# Query and print.
print("========== Similarity Search ==========\n")
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print(results[0], "\n")

# Asynchronous query and print.
print("========== Asynchronous Similarity Search ==========\n")
results = asyncio.run(vector_store.asimilarity_search("When was Nike incorporated?"))
print(results[0], "\n")

# Query and print with similarity score.
# Similarity score varies per LLM provider.
print("========== Similarity Search with Score ==========\n")
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

# Return documents from embedded query similarity search and print 
print("========== Return Documents from Similarity Search ==========\n")
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0], "\n")

# Retrievable that can be invoked since it is a Runnable.
@chain
def retriever(query: str) -> List[Document]:
   return vector_store.similarity_search(query, k=1)

# Use a vector_store (which is not a runnable) to generate a Retriever.
retriever2 = vector_store.as_retriever(
   search_type="similarity",
   search_kwargs={"k": 1}
)

# Query and print.
print("========== Retriever Batch Similarity Search ==========\n")
print(retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
]), "\n")

# Query and print.
print("========== Retriever using Vector Store Batch Similarity Search ==========\n")
print(retriever2.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
]))