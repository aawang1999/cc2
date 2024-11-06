from langchain.text_splitter import RecursiveCharacterTextSplitter


class DataChunkingUtils():
    def __init__(self):
        pass

    def chunk_recursive(self, docs, chunk_size, chunk_overlap=0):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return splitter.split_documents(documents=docs)
