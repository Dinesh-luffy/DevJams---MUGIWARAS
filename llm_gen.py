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
    prompt = f"""
You are a helpful legal assistant. A lawyer asked: "{query}"

Relevant context from legal documents:
{context}

Answer concisely, suggest legal statutes, precedents, or arguments if possible.
"""
    try:
        # The ollama client.generate() method does not accept a max_tokens parameter.
        # It's automatically handled by the model.
        # This line is correct as written in the user's provided code.
        response = client.generate(model="llama2", prompt=prompt)
        
        # Accessing the response content correctly
        # The user's provided code for this part is also slightly off. 
        # The ollama response is a simple dictionary, not an object with a .content attribute.
        # A simple response would be a dictionary like {'model': 'llama2', 'created_at': '...', 'response': '...'}
        
        answer_text = response['response']
        return answer_text.strip()

    except Exception as e:
        print("⚠️ Error generating response:", e)
        return "⚠️ No response from LLaMA model."