from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# save the database in central data folder
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma_db"

def ingest_docs():
    # small fast open embedding source model that runs locally
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # simulated regulatory documents
    docs = [
        #the actual text is indexed and semantically searched
        Document(
            page_content="REG-001: When grid spot prices drop below 0 EUR/MWh, grid operators MUST prioritize charging battery storage systems before curtailing renewable generation.",
            metadata={"source": "EU Grid Directive 2024"}
        ),
        Document(
            page_content="REG-002: If negative spot prices persist for more than 4 consecutive hours, operators are legally required to notify the federal grid agency (BNetzA) via the emergency API.",
            metadata={"source": "BNetzA Guidelines"}
        ),
        Document(
            page_content="REG-003: Under no circumstances should nuclear or coal baseload plants be completely shut down during short-term negative price events due to cold-start costs. Minimum output of 20% must be maintained.",
            metadata={"source": "Baseload Operations Manual"}
        )
    ]
    #embed and persist to chromadb
    print("Initializing vector store and embedding documents...")
    # create the database and persist it to disk
    vectorstore = Chroma.from_documents( 
        documents=docs,
        embedding=embeddings,
        persist_directory=str(DB_PATH)
    )

    #Chroma.from_documents() does: (READ DOCS!!)
    #   passes doc.page_content through the embedding model to compute vector representations
    #   indexes vectors, text, and metadata inside Chroma
    #   saves the underlying database files to the specified directory on disk
    
    print(f"Successfully ingested {len(docs)} regulatory documents into ChromaDB at {DB_PATH}")
    
#entry point
if __name__ == "__main__":
    ingest_docs()