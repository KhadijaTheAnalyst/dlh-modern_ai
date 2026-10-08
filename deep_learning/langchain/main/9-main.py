#!/usr/bin/env python3

load_documents = __import__('8-load_documents').load_documents
split_into_chunks = __import__('9-chunking').split_into_chunks


folder = "ZendeskArticles"
file_pattern = "**/*.md"
docs = load_documents(folder, file_pattern)
docs = sorted(docs, key=lambda doc: doc.metadata["source"])

chunk_size = 1000
chunk_overlap = 200
split_docs = split_into_chunks(docs, chunk_size, chunk_overlap)

print(type(split_docs))
print(f"Total chunks after splitting: {len(split_docs)}")
print(type(split_docs[0]))
print("\n")
print(split_docs[0])
