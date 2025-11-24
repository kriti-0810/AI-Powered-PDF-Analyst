from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

class RAGPipeline:
    def __init__(self, vector_store, embedding_model):
        """
        vector_store = FAISSStore()
        embedding_model = EmbeddingModel()
        """
        load_dotenv()

        self.vector_store = vector_store
        self.embedding_model = embedding_model

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.2
        )

    # ----------------------------------------------------
    # 1) RETRIEVAL
    # ----------------------------------------------------
    def retrieve_context(self, user_query, top_k=5):
        """Embed query and retrieve chunks from FAISS."""
        query_embedding = self.embedding_model.embed_text([user_query])
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Build context string
        formatted_chunks = []
        for r in results:
            formatted_chunks.append(
                f"[Page {r['page_number']} - {r['filename']}]\n{r['content']}"
            )

        context = "\n\n---\n\n".join(formatted_chunks)
        return context, results

    # ----------------------------------------------------
    # 2) ANSWER GENERATION
    # ----------------------------------------------------
    def generate_answer(self, user_query, context):
        """LLM reads the retrieved context + question."""
        prompt = f"""
You are an AI Research Assistant.

Use ONLY the context provided to answer the question.

If the answer is not in the context, say:
"I could not find the answer in the uploaded documents."

--------------------
CONTEXT:
{context}

--------------------
USER QUESTION:
{user_query}

Write the answer with citations in this format:
(Ref: filename, page number)
"""

        response = self.llm.invoke(prompt)
        return response.content

    # ----------------------------------------------------
    # 3) SUMMARY GENERATION
    # ----------------------------------------------------
    def generate_summary(self, context):
        prompt = f"""
Summarize the following document context clearly and concisely:

{context}

Write a clean summary in bullet points.
"""

        response = self.llm.invoke(prompt)
        return response.content

    # ----------------------------------------------------
    # 4) QUIZ GENERATION
    # ----------------------------------------------------
    def generate_quiz(self, context, q_type="mcq"):
        if q_type == "mcq":
            q_format = "Generate 5 MCQs with 4 options each and provide the correct answer at the end."
        else:
            q_format = "Generate 5 short answer questions."

        prompt = f"""
Based on the following context, create a quiz.

CONTEXT:
{context}

{q_format}
"""

        response = self.llm.invoke(prompt)
        return response.content

    # ----------------------------------------------------
    # 5) EXPLAIN MODE
    # ----------------------------------------------------
    def explain_topic(self, context, style="simple"):
        if style == "simple":
            mode = "Explain like I'm 10 years old."
        elif style == "expert":
            mode = "Explain in expert technical detail."
        else:
            mode = "Explain with examples."

        prompt = f"""
Explain the following content.

CONTEXT:
{context}

STYLE: {mode}
"""

        response = self.llm.invoke(prompt)
        return response.content
