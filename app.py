import os
import base64
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Resume Tracking using RAG", page_icon="innomatics_logo.jpg")

# ----------------------------
# Custom CSS
# ----------------------------
CUSTOM_CSS = """
<style>
    .title-box {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        background: linear-gradient(135deg, #1f3c88 0%, #4a69bd 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(31, 60, 136, 0.25);
    }
    .title-box img {
        height: 64px;
        border-radius: 8px;
        background: white;
        padding: 6px;
    }
    .title-box h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .answer-box {
        background-color: #eef1f8;
        border-left: 5px solid #4a69bd;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
    }
    .answer-box h3 {
        margin-top: 0;
        color: #1f3c88;
    }
    .answer-box p, .answer-box li, .answer-box div {
        color: #1a1a2e;
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Title box with logo inside
# ----------------------------
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

logo_b64 = get_base64_image("innomatics_logo.jpg")
logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}">' if logo_b64 else ""

st.markdown(
    f'<div class="title-box">{logo_html}<h1>Resume Tracking using RAG</h1></div>',
    unsafe_allow_html=True
)

resume_folder = r"C:\Users\Aadesh Yogi\Desktop\Projects\Personal Resume RAG\Resumes"


# ----------------------------
# Cached Vector Store Setup
# ----------------------------
@st.cache_resource
def get_vector_store():
    loader = PyPDFDirectoryLoader(resume_folder)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma(embedding_function=embedding, persist_directory='RESUME_REG')

    if vector_store._collection.count() == 0:
        vector_store.add_documents(chunks)

    return vector_store, len(docs), len(chunks)


with st.spinner("Loading and indexing resumes..."):
    vector_store, num_docs, num_chunks = get_vector_store()


# ----------------------------
# Query Input + Button
# ----------------------------
query = st.text_input("Ask a question about the resumes:", placeholder="e.g. What are the skills of Aadesh Yogi?")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a question before searching.")
    else:
        with st.spinner("Retrieving relevant resumes..."):
            retriever = vector_store.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 12, 'lambda_mult': 0.7}
            )
            retrieved_docs = retriever.invoke(query)

            context_text = '\n\n'.join(
                f"['source':{doc.metadata.get('source')}] \n {doc.page_content}"
                for doc in retrieved_docs
            )

            prompt = f"""
You are a professional AI Resume Retrieval Assistant.

You are provided with resume excerpts retrieved from a vector database. These excerpts are your ONLY source of information.

Instructions:
- Read all retrieved excerpts carefully before answering.
- Base every statement strictly on the provided resume text.
- Never fabricate or infer missing information.
- STRICT MATCHING RULE: If the question asks for candidates matching multiple criteria (e.g. multiple skills, multiple conditions joined by "and"), only include a candidate if ALL of the specified criteria are clearly present in their resume excerpt. Do NOT include a candidate who satisfies only some of the criteria.
- If a candidate is missing even one required skill/criterion, exclude them from the answer — do not list them as a partial match.
- If multiple resumes satisfy ALL the criteria, list every relevant candidate.
- If the answer cannot be determined from the retrieved excerpts, or no candidate satisfies every criterion, respond exactly:

"I couldn't find sufficient information in the retrieved resumes."

Response Guidelines:
- Mention the candidate name or resume filename whenever available.
- For each candidate listed, explicitly show which required skills/criteria were found in their resume, to justify why they are a full match.
- Summarize information rather than copying long passages.
- Use bullet points when multiple candidates are returned.
- Keep answers factual and concise.
- If confidence is low because the retrieved context is incomplete, explicitly state that.

Retrieved Resume Context
========================
{context_text}
========================

Question:
{query}

Grounded Answer:
"""

            llm = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4o-mini')
            result = llm.invoke(prompt)

        st.markdown(
            f'<div class="answer-box"><h3>Answer</h3><div>{result.content}</div></div>',
            unsafe_allow_html=True
        )

        with st.expander("View retrieved context"):
            st.text(context_text)