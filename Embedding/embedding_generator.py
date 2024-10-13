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
            agent:
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
    # pc.delete_index("llamaindex-ragathon-demo-index-v2")

    # creating index for now not needed
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
            text = d['text'],
            metadata={
                "timestamp":d['timestamp'],
                "video_id":d['video_id'],
                "agent": d['agent']
            }
        ))

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)


def generate_embedding(agent):
    path=''
    if(agent=='image_captioning'):
        path = 'sources/bahubali/chunks/image_captionings'
    elif agent == 'transcripts':
        path ='sources/bahubali/chunks/transcripts'
    

    data_array = []
    for filename in os.listdir(path):
        print(filename)
        if filename.endswith('.txt'):
            parts = filename.split('_')
            timestamp = parts[1]
            print(timestamp)
            with open(os.path.join(path, filename), 'r') as file:
                text = file.read()
            
            # Create a dictionary and append it to the data_array
            entry = {
                'text': text.strip(),
                'timestamp': timestamp,
                'video_id': 'bahubali',
                'agent': agent  # Replace with the appropriate value if needed
            }
            data_array.append(entry)
    print(data_array)
    create_embeddings(data_array)
    # print(data_array[2]['timestamp'])


    
