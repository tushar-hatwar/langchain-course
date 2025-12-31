import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from azure_env_embed import embeddings
from azure_env import llm

load_dotenv()

print("\n" + "="*80)
print("INITIALIZATION PHASE")
print("="*80)
print(f"Loading environment variables...")
print(f"INDEX_NAME: {os.environ.get('INDEX_NAME')}")

print("\nInitializing Pinecone VectorStore...")
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
print(f"✓ VectorStore initialized: {type(vectorstore).__name__}")

print("\nCreating retriever (k=3)...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"✓ Retriever created: {type(retriever).__name__}")

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    print(f"\n[format_docs] Formatting {len(docs)} documents...")
    for i, doc in enumerate(docs, 1):
        print(f"  Doc {i}: {len(doc.page_content)} chars, metadata: {doc.metadata}")
    formatted = "\n\n".join(doc.page_content for doc in docs)
    print(f"[format_docs] Total formatted length: {len(formatted)} chars")
    return formatted


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    print(f"\n[Step 1] Retrieving documents for query: '{query}'")
    docs = retriever.invoke(query)
    print(f"[Step 1] ✓ Retrieved {len(docs)} documents")

    # Step 2: Format documents into context string
    print(f"\n[Step 2] Formatting documents into context...")
    context = format_docs(docs)
    print(f"[Step 2] ✓ Context created")

    # Step 3: Format the prompt with context and question
    print(f"\n[Step 3] Formatting prompt template...")
    messages = prompt_template.format_messages(context=context, question=query)
    print(f"[Step 3] ✓ Created {len(messages)} message(s)")
    print(f"[Step 3] Message preview: {str(messages[0])[:200]}...")

    # Step 4: Invoke LLM with the formatted messages
    print(f"\n[Step 4] Invoking LLM...")
    response = llm.invoke(messages)
    print(f"[Step 4] ✓ LLM response received: {len(response.content)} chars")

    # Step 5: Return the content
    print(f"\n[Step 5] ✓ Returning response content")
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    print(f"\n[RAW LLM] Sending query directly to LLM without retrieval...")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(f"[RAW LLM] ✓ Response received: {len(result_raw.content)} chars")
    print(f"[RAW LLM] Response type: {type(result_raw).__name__}")
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    print(f"\n[LCEL] Creating chain...")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    print(f"[LCEL] ✓ Chain created: {type(chain_with_lcel).__name__}")
    print(f"\n[LCEL] Invoking chain with question: '{query}'")
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(f"[LCEL] ✓ Chain completed: {len(result_with_lcel)} chars")
    print(f"[LCEL] Result type: {type(result_with_lcel).__name__}")
    print("\nAnswer:")
    print(result_with_lcel)
