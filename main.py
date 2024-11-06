from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_milvus import Milvus
from langchain_ollama import OllamaLLM

from embedding_models import CustomEmbeddingModel
from utils.data_chunking_utils import DataChunkingUtils
from utils.embedding_utils import EmbeddingUtils

chunk_utils = DataChunkingUtils()
embed_utils = EmbeddingUtils()

URI = "./milvus.db"


def main():
    # Load PDF files
    docs = PyPDFDirectoryLoader(
        path="files",
        extract_images=True
    ).load()
    print("Loaded PDF files")

    # Chunk the text; see utils for chunking methods
    chunks = chunk_utils.chunk_recursive(
        docs=docs,
        chunk_size=1024,
        chunk_overlap=100
    )
    print("Chunked text")

    # Load embedding model; see utils for quantization methods
    embedding_model = CustomEmbeddingModel(
        model="dunzhang/stella_en_400M_v5",
        quantization=None
    )
    print("Loaded embedding model")

    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embedding_model,
        connection_args={"uri": URI},
        drop_old=True
    )
    print("Embedded and stored text")

    # Load LLM
    llm = OllamaLLM(
        model="phi3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"]
    )

    # Prompt user for input and create retrieval chain
    query = input("\nQuery: ")
    prompt = hub.pull("rlm/rag-prompt")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    # Output result
    result = qa_chain.invoke({"query": query})


if __name__ == "__main__":
    main()
