#!/usr/bin/env python3

import os
from langchain.schema import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
load_mistral = __import__('1-load_mistral').load_mistral
load_documents = __import__('8-load_documents').load_documents
split_into_chunks = __import__('9-chunking').split_into_chunks
create_embeddings = __import__('10-create_embeddings').create_embeddings
create_faiss_vectorstore = __import__('11-create_faiss_vectorstore').create_faiss_vectorstore
document_answer_chain = __import__('12-document_chain').document_answer_chain
build_history_aware_retriever = __import__('13-history_aware_retriever').build_history_aware_retriever
create_conversational_retrieval_chain = __import__('14-conversational_retrieval_chain').create_conversational_retrieval_chain
create_session_aware_chat = __import__('15-session_aware_chat').create_session_aware_chat

os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"
llm = load_mistral("mistral-small-latest", 0)

folder = "ZendeskArticles"
file_pattern = "**/*.md"
docs = load_documents(folder, file_pattern)
docs = sorted(docs, key=lambda doc: doc.metadata["source"])

chunk_size = 1000
chunk_overlap = 200
split_docs = split_into_chunks(docs, chunk_size, chunk_overlap)
embedding_model = create_embeddings("sentence-transformers/all-MiniLM-L6-v2")

k = 3
_, retriever = create_faiss_vectorstore(split_docs, embedding_model, k)

human_msg_retriever = "{input}\n\nFormulate a search query based on the conversation above."
_, history_aware_retriever = build_history_aware_retriever(
    llm, retriever, human_msg_retriever
)

system_msg = "Based on the context below, provide a one-sentence answer to the user's question*:\n\n{context}"
human_msg = "{input}"
_, qa_document_chain = document_answer_chain(llm, system_msg, human_msg)

qa_chain = create_conversational_retrieval_chain(history_aware_retriever, qa_document_chain)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

session_aware_qa_chain = create_session_aware_chat(
    qa_chain,
    get_session_history,
    input_key="input",
    history_key="chat_history",
    output_key="answer",
)

print("RAG Q&A Chatbot is ready! Type 'exit' to stop.\n")
session_id = "holbie_1"

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    response = session_aware_qa_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    print("Bot:", response)
