#!/usr/bin/env python3

load_documents = __import__('8-load_documents').load_documents
split_into_chunks = __import__('9-chunking').split_into_chunks
create_embeddings = __import__('10-create_embeddings').create_embeddings


folder = "ZendeskArticles"
file_pattern = "**/*.md"
docs = load_documents(folder, file_pattern)
docs = sorted(docs, key=lambda doc: doc.metadata["source"])

chunk_size = 1000
chunk_overlap = 200
split_docs = split_into_chunks(docs, chunk_size, chunk_overlap)

embedding_model = create_embeddings("sentence-transformers/all-MiniLM-L6-v2")

sample_text = split_docs[0].page_content
print(sample_text)
print("\n")

embedding_vector = embedding_model.embed_documents([sample_text])
print(embedding_vector[0][:10])
print(len(embedding_vector[0]))
print(type(embedding_model))
