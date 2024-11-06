from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from utils.embedding_utils import EmbeddingUtils

embed_utils = EmbeddingUtils()


class CustomEmbeddingModel:
    def __init__(self, model, quantization=None):
        self.model = SentenceTransformer(
            model_name_or_path=model,
            trust_remote_code=True,
            device="cpu",
            config_kwargs={
                "use_memory_efficient_attention": False,
                "unpad_inputs": False
            }
        )
        self.quantization = quantization

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = np.array(
            np.array([self.model.encode(text) for text in texts])
        )

        if self.quantization:
            embeddings = embed_utils.create_quantized_embeddings(
                embeddings=embeddings,
                quantization=self.quantization
            )
            print(f"Docs {self.quantization} quantized")

        print(f"Doc embed dims: {embeddings.shape}")
        print(f"Doc embed vals: {embeddings.dtype}")
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        encoded_query = np.array([self.model.encode(text) for text in [query]])

        if self.quantization:
            encoded_query = embed_utils.create_quantized_embeddings(
                embeddings=encoded_query,
                quantization=self.quantization
            )
            print(f"Query {self.quantization} quantized")

        print(f"Query embed dims: {encoded_query.shape}")
        print(f"Query embed vals: {encoded_query.dtype}")
        print(f"Query embed: {encoded_query}")
        return encoded_query.squeeze().tolist()
