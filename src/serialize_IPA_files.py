from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import threading
from resource_monitoring import monitor_usage

load_dotenv()

file_origin = os.environ.get("FILE_ORIGIN")

# Path to destination folder for serialized vector store
ipa_faiss_path = file_origin + os.environ.get("IPA_PATH")

# Path to easily-loaded resources
knowledge_path_easy = file_origin + 'Easy_Parse/'

# Path to difficult-to-parse (IPA) resources
knowledge_path_hard = file_origin + 'Hard_Parse/'

# Add knowledge_path_hard or any other sources to the list here
knowledge_source_list = [knowledge_path_easy]

# Start monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_usage)
monitor_thread.daemon = True  # Allow the thread to exit when the main program exits
monitor_thread.start()

def main():
  # IPA retriever self-contained tool
  if ipa_faiss_path.removeprefix(file_origin) not in os.listdir(file_origin):
    for knowledge_path in knowledge_source_list:
      print(os.listdir(knowledge_path))
      ipa_docs = []
      # Load the pdf documents across all source paths
      for fn in os.listdir(knowledge_path):
        file_path = os.path.join(knowledge_path, fn)
        print(fn)
        if fn[-3:] == 'pdf':
          loader = PyPDFLoader(file_path, extract_images = False)
          documents = loader.load()
          # Correcting reading issues and splitting into smaller chunks
          for doc in documents:
            doc.page_content = doc.page_content.replace('\n', ' ')
          split_documents = TokenTextSplitter(model_name = 'gpt-4o', chunk_size=300, chunk_overlap=30).split_documents(documents)
          ipa_docs.extend(split_documents)

    # Create vectorstore from previously-compiled document list
    ipa_db = FAISS.from_documents(ipa_docs, OpenAIEmbeddings())
    print(ipa_db.index.ntotal)
    ipa_db.save_local(ipa_faiss_path)

if __name__ == "__main__":
  main()