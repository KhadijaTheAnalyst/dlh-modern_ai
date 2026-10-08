#!/usr/bin/env python3

load_documents = __import__('8-load_documents').load_documents


folder = "ZendeskArticles"
file_pattern = "**/*.md"
docs = load_documents(folder, file_pattern)
docs = sorted(docs, key=lambda doc: doc.metadata["source"])

print(type(docs))
print(f"Loaded {len(docs)} documents.")
print(type(docs[0]))
print("\n")
print(docs[0])
