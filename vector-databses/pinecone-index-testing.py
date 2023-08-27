import pinecone

pinecone.init(api_key="51d35fe6-6d32-405d-81f0-fb9f42f3b2bc", environment="us-west4-gcp-free")

#create the index
#pinecone.create_index("quickstart", dimension=8, metric="euclidean")

#list the index
pinecone.list_indexes()


# connect to the index
index = pinecone.Index("quickstart")

#fill the index with data
index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])


# print stats about the index
print(index.describe_index_stats())

# query the index and get similar vectors
print(index.query(
    vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    top_k=3,
   include_values=True
))

# delete the index
pinecone.delete_index('quickstart')

print("index successfully deleted")
