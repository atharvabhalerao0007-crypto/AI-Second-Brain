import os
import streamlit as st
import numpy as np

from config.settings import (
    DOCUMENT_PATH,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
    GROQ_API_KEY
)

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import Embeddings
from src.vector_store import VectorStore
from src.llm import LLMWrapper
from src.rag_pipeline import RAGPipeline
from features.document_analytics import get_document_stats

from features.knowledge_graph import build_graph, draw_graph
from streamlit.components.v1 import html

from streamlit_mic_recorder import mic_recorder
from utils.voice import speak_text
import whisper
import tempfile


st.set_page_config(page_title="AI Second Brain", layout="wide")
@st.cache_resource
def load_whisper():
    import whisper
    return whisper.load_model("base")

whisper_model = load_whisper()

# -----------------------
# Custom UI Styling
# -----------------------

st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* Title */
h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 700;
}

/* Headers */
h2, h3 {
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* Buttons */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background-color: #161b22;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
}

/* Input fields */
input, textarea {
    background-color: #0e1117 !important;
    color: #ffffff !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #161b22 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    padding: 10px;
    border-radius: 10px;
}

/* Code blocks */
code {
    color: #58a6ff;
}

</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Second Brain")
st.caption("AI-powered document assistant with RAG, Knowledge Graphs, Quiz Generation and Insights.")
st.divider()

# -----------------------
# Sidebar: Feature Selection
# -----------------------
feature = st.sidebar.selectbox(
    "Select Feature",
    [
        "RAG QA",
        "Semantic Search",
        "Notes Generator",
        "Quiz Generator",
        "Knowledge Graph",
        "Document Summary",
        "AI Insights",
        "Document Analytics"
    ]
)

# -----------------------
# Sidebar: Upload PDF
# -----------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs(DOCUMENT_PATH, exist_ok=True)

    for uploaded_file in uploaded_files:

        file_path = os.path.join(DOCUMENT_PATH, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success("Documents uploaded successfully!")


# -----------------------
# Load Documents
# -----------------------
loader = DocumentLoader(DOCUMENT_PATH)
docs = loader.load_documents()

if not docs:
    st.warning("No documents found in the documents folder.")
    st.stop()


# -----------------------
# Initialize Pipeline (Cached)
# -----------------------

@st.cache_resource
def initialize_pipeline():

    loader = DocumentLoader(DOCUMENT_PATH)
    docs = loader.load_documents()

    if not docs:
        return None, None

    splitter = TextSplitter(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    chunks = [c for d in docs for c in splitter.split_text(d)]

    embedder = Embeddings(model_name=EMBEDDING_MODEL)
    embeddings = embedder.get_embeddings(chunks)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    store_file = os.path.join(VECTOR_DB_PATH, "store.pkl")

    store = VectorStore(dim=len(embeddings[0]), store_path=store_file)

    store.add_vectors(np.array(embeddings), chunks)

    return chunks, store


chunks, store = initialize_pipeline()

if chunks is None:
    st.warning("No documents found in the documents folder.")
    st.stop()


# -----------------------
# Initialize LLM & RAG
# -----------------------
llm = LLMWrapper(api_key=GROQ_API_KEY, model=LLM_MODEL)

rag = RAGPipeline(vector_store=store, llm=llm)


# -----------------------
# Feature: RAG QA
# -----------------------
if feature == "RAG QA":

    st.header("RAG Question Answering")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    # Page filter
    page_filter = st.number_input("Search within page (optional)", min_value=1, step=1)

    # Chat input
    user_question = st.chat_input("Ask something about your document...")

st.write("🎤 Or ask using voice")

voice_data = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True
)
import tempfile

# Convert voice to text
if voice_data:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(voice_data["bytes"])
        audio_path = f.name

    result = whisper_model.transcribe(audio_path)

    user_question = result["text"]

    st.info(f"Voice Question: {user_question}")

if user_question:

        with st.spinner("Generating answer..."):

            # If page filter is used
            if page_filter:

                filtered_chunks = [c for c in chunks if f"Page {page_filter}" in c]

                if filtered_chunks:
                    context = " ".join(filtered_chunks[:3])

                    answer = llm.generate_text(
                        f"Answer the question using only this content:\n{context}\n\nQuestion: {user_question}"
                    )

                    sources = filtered_chunks

                else:
                    answer = "No content found for that page."
                    sources = []

            # Normal RAG search
            else:

                answer, sources = rag.answer_question(
    question=user_question,
    top_k=5
)
            from gtts import gTTS
import io

# Ensure answer exists
if answer and isinstance(answer, str):

    try:
        tts = gTTS(text=answer[:500], lang="en")  # limit length

        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)

        st.audio(audio_bytes.getvalue(), format="audio/mp3")

    except Exception as e:
        st.warning("Voice response unavailable.")
        st.error(e)
        
        # Save chat
st.session_state.chat_history.append(
            {"question": user_question, "answer": answer}
        )

    # Display chat history
for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

    # -----------------------
# Conversation Summary
# -----------------------

if st.button("Summarize Conversation"):

    conversation = ""

    for chat in st.session_state.chat_history:
        conversation += f"Q: {chat['question']}\nA: {chat['answer']}\n\n"

    summary_prompt = f"""
Summarize this conversation clearly:

{conversation}
"""

    summary = llm.generate_text(summary_prompt)

    st.subheader("Conversation Summary")

    st.success(summary)


    # Show sources
    if user_question:

        st.subheader("Sources")

        unique_sources = list(dict.fromkeys(sources))

    for i, src in enumerate(unique_sources, 1):

     page_info = "Unknown Page"

    if "Page" in src:
        page_info = src.split("Page")[-1].strip()

    with st.expander(f"Source {i} | Page {page_info}"):

        st.write(src)
   

# -----------------------
# Feature: Semantic Search
# -----------------------
elif feature == "Semantic Search":

    st.header("Semantic Search")

    query = st.text_input("Enter search query:")

    top_k = st.slider("Top results", min_value=1, max_value=10, value=3)

    if query:

        question_vec = np.array([llm.get_embedding(query)])

        results = store.search(question_vec, top_k=top_k)

        st.subheader("Top Relevant Chunks:")

        for i, chunk in enumerate(results, 1):
            st.write(f"{i}. {chunk[:300]}...")


# -----------------------
# Feature: Notes Generator
# -----------------------
elif feature == "Notes Generator":

    st.header("Notes Generator")

    topic = st.text_input("Enter topic for notes:")

    if topic:

        prompt = f"Generate concise notes on the topic: {topic}\nUse the documents as reference."

        notes = llm.generate_text(prompt)

        st.subheader("Generated Notes:")

        st.success(notes)

# -----------------------
# Feature: Quiz Generator
# -----------------------
elif feature == "Quiz Generator":

    st.header("AI Quiz")

    if "score" not in st.session_state:
        st.session_state.score = 0

    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None

    topic = st.text_input("Enter topic for quiz")

    if st.button("Generate Question") and topic:

        prompt = f"""
Create ONE multiple choice question about {topic}.

Format exactly like this:

Question: ...
A) ...
B) ...
C) ...
D) ...
Answer: A
"""

        quiz = llm.generate_text(prompt)

        lines = quiz.split("\n")

        question = ""
        options = {}
        answer = ""

        for line in lines:

            line = line.strip()

            if line.startswith("Question"):
                question = line

            elif line.startswith("A)"):
                options["A"] = line

            elif line.startswith("B)"):
                options["B"] = line

            elif line.startswith("C)"):
                options["C"] = line

            elif line.startswith("D)"):
                options["D"] = line

            elif line.startswith("Answer"):
                answer = line.split(":")[1].strip()

        st.session_state.quiz_data = {
            "question": question,
            "options": options,
            "answer": answer
        }

    if st.session_state.quiz_data:

        q = st.session_state.quiz_data

        st.write(q["question"])

        user_choice = st.radio(
            "Choose your answer:",
            list(q["options"].keys()),
            format_func=lambda x: q["options"][x]
        )

        if st.button("Submit Answer"):

            if user_choice == q["answer"]:

                st.success("Correct! 🎉")

                st.session_state.score += 1

            else:

                st.error(f"Wrong! Correct answer: {q['options'][q['answer']]}")

        st.subheader(f"Score: {st.session_state.score}")



# -----------------------
# Feature: Knowledge Graph
# -----------------------
elif feature == "Knowledge Graph":

    st.header("Knowledge Graph")

    # -----------------------
    # Session State
    # -----------------------

    if "kg_graph" not in st.session_state:
        st.session_state.kg_graph = None

    if "kg_nodes" not in st.session_state:
        st.session_state.kg_nodes = []

    if "selected_concept" not in st.session_state:
        st.session_state.selected_concept = None

    if "concept_explanation" not in st.session_state:
        st.session_state.concept_explanation = None

    # -----------------------
    # Generate Graph
    # -----------------------

    if st.button("Generate Graph"):

        G = build_graph(chunks)

        if len(G.nodes) == 0:

            st.warning("No entities found in documents.")

        else:

            st.session_state.kg_graph = G
            st.session_state.kg_nodes = list(G.nodes)[:15]

    # -----------------------
    # Show Graph
    # -----------------------

    if st.session_state.kg_graph:

        import matplotlib.pyplot as plt
        import networkx as nx

        subgraph = st.session_state.kg_graph.subgraph(st.session_state.kg_nodes)

        plt.figure(figsize=(10,7))

        pos = nx.spring_layout(subgraph, k=0.6)

        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=2500,
            font_size=10,
            edge_color="gray"
        )

        st.pyplot(plt)

        # -----------------------
        # Interactive Graph (Pyvis)
        # -----------------------

        st.subheader("Interactive Knowledge Graph")

        interactive_html = draw_graph(subgraph)

        html(interactive_html, height=650)
        
        # -----------------------
        # Concept Explorer
        # -----------------------

        st.subheader("Explore a Concept")

        concept = st.selectbox(
            "Select a concept:",
            st.session_state.kg_nodes
        )

        if st.button("Explain Concept"):

            prompt = f"""
Explain the concept '{concept}' clearly.

Include:
- Definition
- Why it is important
- Example
"""

            explanation = llm.generate_text(prompt)

            st.session_state.concept_explanation = explanation
            st.session_state.selected_concept = concept

    # -----------------------
    # Display Explanation
    # -----------------------

    if st.session_state.concept_explanation:

        st.subheader(f"Explanation: {st.session_state.selected_concept}")

        st.success(st.session_state.concept_explanation)

# -----------------------
# Feature: Document Summary
# -----------------------
elif feature == "Document Summary":

    st.header("AI Document Summary")

    if st.button("Generate Summary"):

        with st.spinner("Summarizing document..."):

            context = " ".join(chunks[:5])

            prompt = f"""
Summarize the following document content into clear bullet points.

Content:
{context}

Provide a concise summary.
"""

            summary = llm.generate_text(prompt)

        st.subheader("Summary")

        st.success(summary)

# -----------------------
# Feature: Knowledge Map
# -----------------------
elif feature == "Knowledge Map":

    st.header("AI Knowledge Map")

    if st.button("Generate Knowledge Map"):

        context = " ".join(chunks[:10])

        prompt = f"""
Extract the main topics and subtopics from the following text.

Format like a knowledge tree.

Example format:

Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing

Text:
{context}
"""

        result = llm.generate_text(prompt)

        st.subheader("Concept Map")

        st.write(result)

# -----------------------
# Feature: AI Insights
# -----------------------

elif feature == "AI Insights":

    st.header("📊 AI Document Insights")

    if st.button("Generate Insights"):

        with st.spinner("Analyzing document..."):

            context = " ".join(chunks[:5])

            prompt = f"""
Analyze the following document and extract useful insights.

Content:
{context}

Provide:

1. Key Topics
2. Important Entities (people, companies, technologies)
3. Key Takeaways
4. Difficulty Level (Beginner / Intermediate / Advanced)

Format clearly.
"""

            insights = llm.generate_text(prompt)

        st.success(insights)

# ---------------------------
# Feature: Document Analytics
# ---------------------------
elif feature == "Document Analytics":

    import pandas as pd
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    st.header("📊 Document Analytics Dashboard")

    stats = get_document_stats(" ".join(chunks), chunks)

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Pages", stats["pages"])
    col2.metric("🧩 Chunks", stats["chunks"])
    col3.metric("🏷 Entities", len(stats["entities"]))
    col4.metric("⏱ Reading Time", f"{stats['reading_time']} min")

    st.divider()

    # -------------------
    # Top Keywords Table
    # -------------------

    st.subheader("🔑 Top Keywords")

    keyword_df = pd.DataFrame(stats["keywords"], columns=["Keyword", "Frequency"])

    st.dataframe(keyword_df, use_container_width=True)

    # -------------------
    # Keyword Chart
    # -------------------

    st.subheader("📊 Keyword Frequency")

    fig, ax = plt.subplots()

    ax.bar(keyword_df["Keyword"], keyword_df["Frequency"])

    ax.set_xlabel("Keyword")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # -------------------
    # Word Cloud
    # -------------------

    st.subheader("☁ Word Cloud")

    text = " ".join([word for word, count in stats["keywords"] for i in range(count)])

    wc = WordCloud(width=800, height=400, background_color="black").generate(text)

    fig_wc, ax_wc = plt.subplots()

    ax_wc.imshow(wc)
    ax_wc.axis("off")

    st.pyplot(fig_wc)

    # -------------------
    # Entity Distribution
    # -------------------

    st.subheader("🏷 Entity Distribution")

    entity_df = pd.DataFrame(stats["entities"], columns=["Entity", "Type"])

    entity_counts = entity_df["Type"].value_counts()

    fig2, ax2 = plt.subplots()

    ax2.bar(entity_counts.index, entity_counts.values)

    ax2.set_xlabel("Entity Type")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)