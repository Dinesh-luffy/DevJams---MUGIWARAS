# llm_gen.py
import ollama

# Make sure Ollama server is running
client = ollama.Client(host="http://127.0.0.1:11434")

def generate_answer(query, context=""):
    """
    Generate a legal response using LLaMA2 with retrieved context.
    
    Args:
        query (str): Lawyer's question.
        context (str): Retrieved context from FAISS/RAG pipeline.
    
    Returns:
        str: Model's response.
    """
    if context:
        prompt = f"""
You are a helpful legal assistant. Your task is to provide an answer based ONLY on the provided legal context.
If the information is not present in the context, state that you cannot find a relevant answer.

Relevant context from legal documents:
{context}

A lawyer asked: "{query}"

Answer concisely, citing legal statutes, precedents, or arguments from the provided context if possible.
"""
    else:
        prompt = f"""
You are a helpful legal assistant for the Indian legal system. When asked about a number, prioritize its meaning as a legal section (e.g., Indian Penal Code) over any other meaning.

A lawyer asked a general legal question: "{query}"

Answer the question concisely based on your general knowledge.
"""
    try:
        response = client.generate(model="llama2", prompt=prompt)
        answer_text = response.get('response', '') 
        return answer_text.strip()

    except Exception as e:
        print("⚠️ Error generating response:", e)
        return "⚠️ No response from LLaMA model."