print(-1)
from transformers.dynamic_module_utils import get_imports
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not certifi.is_available() and "certifi" in imports:
        imports.remove("certifi")
    
    return imports
import textract
from transformers import GPT2TokenizerFast
print(0)
# import certifi
# fix the imports

import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import OllamaEmbeddings
import requests
import os
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Load model directly

# create model
LLM = "mistral:7b"

llm_model = Ollama(model=LLM)
print(1)


def process_pdf_folder(pdf_folder_name,txt_folder_name):
    # # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    print(2)
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    # Array to hold all chunks
    all_chunks = []

    # Iterate over all files in the folder
    for filename in os.listdir(pdf_folder_name):
        # Only process PDF files
        if filename.endswith(".pdf"):
            # Full path to the file
            filepath = os.path.join(pdf_folder_name, filename)

            # Extract text from the PDF file
            doc = textract.process(filepath)

            # Write the extracted text to a .txt file
            txt_filename = filename.replace(".pdf", ".txt")
            txt_filepath = os.path.join(txt_folder_name, txt_filename)

            with open(txt_filepath, 'w') as f:
                f.write(doc.decode('utf-8'))

            # Read the .txt file
            with open(txt_filepath, 'r') as f:
                text = f.read()

            # Split the text into chunks
            chunks = text_splitter.create_documents([text])

            # Add chunks to the array
            all_chunks.append(chunks)

    # Return the array of chunks
    return all_chunks

# Create embeddings 
# os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODEL = 'mxbai-embed-large'
embeddings = OllamaEmbeddings(model=MODEL)

# Store embeddings to vector db
all_chunks = process_pdf_folder("./pdf", "./text")
db =  FAISS.from_documents(all_chunks[0], embeddings) 
for chunk in all_chunks[1:]:
    db_temp = FAISS.from_documents(chunk, embeddings)
    db.merge_from(db_temp)   
chat_history = []

# ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
qa = ConversationalRetrievalChain.from_llm(llm_model, db.as_retriever())

while True:
    # query_start_time = time.time()

    # Get user query
    query = input("Enter a query (type 'exit' to quit): ")
    if query.lower() == "exit":      
        break

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    print(result['answer'])
    # query_end_time = time.time()
    # print(f"Query processed in {query_end_time - query_start_time:.2f} seconds.")



print("Exited!!!")
