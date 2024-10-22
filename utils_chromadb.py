import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
import uuid

class ChromaVDB:
    def __init__(self, persistant=True, chromadb_path='./chroma_db'):
        self.text_splitter = CharacterTextSplitter(chunk_size=500, separator= " \n", chunk_overlap=50)
        self.chroma_db_path = chromadb_path

        if persistant:
            self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        else:
            self.client = chromadb.Client()

        self.col = self.client.get_or_create_collection(name="main")

    def read_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text

    def add_to_db(self, file_path):
        if file_path[-4:] == ".pdf":
            text = self.read_pdf(file_path)
            with open('temp.txt','w') as ff:
                ff.write(text)
            text_file = 'temp.txt'
        else: text_file = file_path
        loader = TextLoader(text_file)
        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)
        docs_ = [doc.page_content for doc in docs]
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        metadatas = [ {"type":"pdf","source":file_path } for _ in range(len(docs)) ]

        self.col.add(
            documents=docs_,
            ids=ids,
            metadatas=metadatas
        )

        print('total docs',self.col.count())

    def retrieve_data(self, query, top_k=7):
        if not self.client:
            raise ValueError("The database is not initialized.")
        results = self.col.query(query_texts=[query], n_results=top_k)
        return results

# Example usage:
if __name__ == "__main__":
    chatbot = ChromaVDB(persistant=False)

    # Load and chunk text file
    # Initialize and add documents to the database
    chatbot.add_to_db('change-management.pdf')

    # Query the database
    query = "Tell me about leadership styles"
    results = chatbot.retrieve_data(query,top_k=3)
    context = '\n\n'.join([doc[0] for doc in results['documents']])
    print(context)
