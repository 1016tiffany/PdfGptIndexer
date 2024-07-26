import textract
import time
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from unittest.mock import patch
from langchain.prompts import PromptTemplate
from huggingface_hub import login
from pathlib import Path

login(token="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")


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.dynamic_module_utils import get_imports

from langchain_community.embeddings import OllamaEmbeddings

import os
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#mistral_models_path = Path.home().joinpath(local_path, '7B-Instruct-v0.3')




# Load model directly
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(2)
# fix the imports
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# create model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
pipe = pipeline("text-generation", model=model_name)
print(1)


with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    llm_model = AutoModelForCausalLM.from_pretrained(
             model_name,
             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
             device_map='auto' if torch.cuda.is_available() else None,
             trust_remote_code=True,
             token="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"
             )
    tokenizer = AutoTokenizer.from_pretrained(
                 model_name,
                 trust_remote_code=True,
                 token="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"
             )
print(type(llm_model))
#llm_model = AutoModelForCausalLM.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)


# Test
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
print(3)
# Generate
generate_ids = llm_model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(4)
def process_pdf_folder(pdf_folder_name,txt_folder_name):
    # # Initialize tokenizer

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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODEL = 'llama3'
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
    query_start_time = time.time()

    # Get user query
    query = input("Enter a query (type 'exit' to quit): ")
    if query.lower() == "exit":      
        break

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    print(result['answer'])
    query_end_time = time.time()
    print(f"Query processed in {query_end_time - query_start_time:.2f} seconds.")



print("Exited!!!")
