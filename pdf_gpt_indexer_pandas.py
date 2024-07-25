import time
import textract
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import pandas as pd

def process_pdf_folder(pdf_folder_name, txt_folder_name):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 24,
        length_function = count_tokens,
    )
    all_chunks = []

    for filename in os.listdir(pdf_folder_name):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder_name, filename)
            doc = textract.process(filepath)
            txt_filename = filename.replace(".pdf", ".txt")
            txt_filepath = os.path.join(txt_folder_name, txt_filename)
            with open(txt_filepath, 'w') as f:
                f.write(doc.decode('utf-8'))
            with open(txt_filepath, 'r') as f:
                text = f.read()
            chunks = text_splitter.create_documents([text])
            all_chunks.append(chunks)
    return all_chunks

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
MODEL = 'llama3'
embeddings = OllamaEmbeddings(model=MODEL)
all_chunks = process_pdf_folder("./pdf", "./text")
db = FAISS.from_documents(all_chunks[0], embeddings)
for chunk in all_chunks[1:]:
    db_temp = FAISS.from_documents(chunk, embeddings)
    db.merge_from(db_temp)

chat_history = []
results_df = pd.DataFrame(columns=["Question", "Output", "Runtime(s)"])
llm_model = Ollama(model=MODEL)
qa = ConversationalRetrievalChain.from_llm(llm_model, db.as_retriever())

while True:
    query_start_time = time.time()
    query = input("Enter a query (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    query_end_time = time.time()
    runtime = query_end_time - query_start_time
    results_df = results_df.append({"Question": query, "Output": result['answer'], "Runtime(s)": f"{runtime:.2f}"}, ignore_index=True)

print(results_df)
