import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
import os
import pytesseract
import time
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# --- Rate Limiter ---


class RateLimiter:
    def __init__(self, max_calls, per_seconds):
        self.max_calls = max_calls
        self.per = per_seconds
        self.calls = deque()

    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] > self.per:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_for = self.per - (now - self.calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.calls.append(time.time())


# Example: 8 requests per 60 seconds
limiter = RateLimiter(8, 60)


def safe_generate(model, prompt):
    limiter.wait()
    return model.generate_content(prompt)


# --- Initialize API ---
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "❌ GOOGLE_API_KEY not found in .env file. Please add it and restart the app.")
    st.stop()

genai.configure(api_key=api_key)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.istockphoto.com/id/1501192520/video/a-visual-generative-ai-background-circuit.jpg?s=640x640&k=20&c=bxYPbAklJ8a1CxqpkYXXN4eFsnkao1YomOfHFu-68gs=");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        background-color: #000000
    }
    .stTextInput > div > div > input {
        background-color: rgb(0,0,0) !important;
        color: rgb(255,255,255) !important;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: rgb(0,0,0) !important;
        color: rgb(255,255,255) !important;
    }
    .stButton button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .st-bb {
        background-color: #bbdefb !important;      
    }
    div[data-testid="stTabs"] button {
        background-color: #1976d2 !important;
        color: #fff !important;
        font-size: 1.2rem !important;
        padding: 12px 32px !important;
        border-radius: 10px 10px 0 0 !important;
        font-weight: bold !important;
        margin-right: 8px !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #388e3c !important;
        color: #fff !important;
        border-bottom: 4px solid #d84315 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 style="color:rgb(139,0,0);text-align:center;background-color:rgba(225,255,250,0.7);border-radius:20px;font-weight: italic;">📚 StudyMate: Conversational Q&A from Academic PDFs</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='
    background-color:rgba(0,0,0,0.6);
    color:rgb(225,250,250);
    padding:16px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    text-align:center;
'>
    📥 Upload your <span style="color:#388e3c;">study PDF</span>, ask any question, and get answers <span style="color:#d84315;">grounded in your material!</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "pdf_history" not in st.session_state:
    st.session_state.pdf_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2, tab3 = st.tabs(
    [
        "📄 PDF Q&A",
        "💬 General Chatbot",
        "🕶️ Text-to-Image"
    ]
)


def generate_image_colab(prompt, negative_prompt, api_url):
    data = {"prompt": prompt, "negative_prompt": negative_prompt}
    response = requests.post(api_url + "/generate", json=data)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None


with tab1:
    uploaded_files = st.file_uploader(
        "", type="pdf", accept_multiple_files=True)
    pdf_texts = []
    pdf_names = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text
                else:
                    # If no text found, try OCR on the page image
                    pix = page.get_pixmap()
                    img = Image.open(BytesIO(pix.tobytes()))
                    # Uncomment and set the path below if needed:
                    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text
            pdf_texts.append(text)
            pdf_names.append(uploaded_file.name)
        st.markdown("### Select PDF to query:")
        selected_pdf = st.selectbox("", pdf_names)
        selected_index = pdf_names.index(selected_pdf)
        selected_text = pdf_texts[selected_index]

        words = selected_text.split()
        chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 300)]
        if not chunks:
            st.error("PDF is empty or could not be processed.")
        else:
            embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            embeddings = embedder.encode(chunks)
            embeddings = np.array(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            st.markdown("### Ask a question from your selected PDF:")
            question = st.text_area("", key="pdf_question")
            if question:
                question_embedding = embedder.encode([question])
                question_embedding = np.array(question_embedding)
                if question_embedding.ndim == 1:
                    question_embedding = question_embedding.reshape(1, -1)
                if question_embedding.shape[1] != dimension:
                    st.error(
                        "Embedding dimension mismatch. Please check your PDF or question.")
                else:
                    D, I = index.search(question_embedding,
                                        k=min(3, len(chunks)))
                    retrieved_chunks = [chunks[i] for i in I[0]]
                    context = "\n".join(retrieved_chunks)

                    # LLM answer generation (Gemini) - Using rate limiter
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    prompt = f"""You are a helpful academic assistant.
                    Use the following context extracted from the student's PDF to answer the question accurately.
                    If the answer is not found in the context, clearly say: "This information is not available in the uploaded PDF."
                    Context:
                   {context}
                   Question: {question}
                   Answer in a clear, structured way:"""
                    with st.spinner("Generating answer..."):
                        response = safe_generate(model, prompt)
                    st.success("🎯 **Answer:**")
                    st.markdown(
                        f"<div style='background-color:rgb(0,0,0);padding:10px;border-radius:8px;color:rgb(255,255,255);'>{response.text}</div>", unsafe_allow_html=True)

                    # Add to PDF Q&A history
                    st.session_state.pdf_history.append(
                        (question, response.text))

                    # Download button for answer
                    st.download_button(
                        "Download Answer", response.text, file_name="pdf_answer.txt")

            # Show PDF Q&A history
            if st.session_state.pdf_history:
                st.markdown("#### Q&A History")
                for q, a in st.session_state.pdf_history:
                    st.markdown(f"<b>Q:</b> {q}", unsafe_allow_html=True)
                    st.markdown(f"<b>A:</b> {a}", unsafe_allow_html=True)
                    st.markdown("---")

with tab2:
    st.markdown('<h4 style="color:rgb(0,0,0);background-color:rgba(255,160,122,0.2);">Ask anything (General Chatbot):</h4>', unsafe_allow_html=True)
    user_input = st.text_area("", key="general_question")
    model = genai.GenerativeModel("gemini-2.5-flash")
    if user_input:
        with st.spinner("💡 Generating answer..."):
            response = safe_generate(model, user_input)
        st.success("🤖 **Answer:**")
        st.markdown(
            f"<div style='background-color:#fffde7;padding:10px;border-radius:8px;color:#6d4c41;'>{response.text}</div>", unsafe_allow_html=True)

        st.session_state.chat_history.append((user_input, response.text))

        # Download button for chatbot answer
        st.download_button("Download Answer", response.text,
                           file_name="chatbot_answer.txt")

    # Show chatbot history
    if st.session_state.chat_history:
        st.markdown("#### Chatbot History")
        for q, a in st.session_state.chat_history:
            st.markdown(f"<b>You:</b> {q}", unsafe_allow_html=True)
            st.markdown(f"<b>Bot:</b> {a}", unsafe_allow_html=True)
            st.markdown("---")

with tab3:
    st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🕶️ High-Fashion Portrait Generator</h1>",
                unsafe_allow_html=True)
    st.divider()
    prompt = st.text_input("Enter your image prompt:")
    negative_prompt = st.text_input(
        "Enter negative prompt (optional):", value="")

    # Set your Colab ngrok API URL here
    api_url = st.text_input(
        "Colab API URL (e.g., http://xxxx-xx-xx-xx-xx.ngrok-free.app)")

    if st.button("✨ Generate Portrait", key="portrait_button"):
        with st.spinner("🧠 Crafting your high-fashion masterpiece..."):
            image = generate_image_colab(prompt, negative_prompt, api_url)
            if image:
                st.image(image, caption="🎨 Your Generated Portrait",
                         use_column_width=True)
                image.save("chroma.png")
                st.success("✅ Portrait generated and saved as chroma.png!")
            else:
                st.error("Failed to generate image from Colab API.")
