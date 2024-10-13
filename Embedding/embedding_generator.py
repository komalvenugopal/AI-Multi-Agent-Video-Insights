import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import TextNode



'''
    data should be an array of json Objects
    [
        {
            text:
            timestamp:
            video_id:
        }
    ]

'''



def create_embeddings(data):
    # hugginface model for embedding
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5")

    #pinconeAPI
    os.environ[
        "PINECONE_API_KEY"
    ] = "aafc89a7-408c-4bbc-8193-58e103949c98"

    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)

    # delete if needed
    pc.delete_index("llamaindex-ragathon-demo-index-v2")

    #creating index for now not needed
    # pc.create_index(
    #     "llamaindex-ragathon-demo-index-v2",
    #     dimension=768, #dimesions for the embedding
    #     metric="euclidean",
    #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # )

    pinecone_index = pc.Index("llamaindex-ragathon-demo-index-v2")

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace="test_05_14"
    )

    nodes =[]
    for d in data:
        nodes.append(TextNode(
            text = d.text,
            metadata={
                "timestamp":d.timestamp
                "video_id":d.video_id
            }
        ))

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)


