from sentence_transformers.quantization import quantize_embeddings


class EmbeddingUtils():
    def __init__(self):
        pass

    # Quantizes embeddings: either None, "int8", or "binary"
    def create_quantized_embeddings(self, embeddings, quantization=None):
        quantized = quantize_embeddings(
            embeddings=embeddings,
            precision=quantization
        )

        return quantized
