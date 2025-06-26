import os
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import base64
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain

# --- Load env ---
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# --- Utility: Extract text from PDFs and chunk ---
def load_and_chunk_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

# --- Combine and embed both resume + personal ---
documents = load_and_chunk_pdf("resume.pdf") + load_and_chunk_pdf("mahitha.pdf")
vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# --- Prompt ---
custom_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are an intelligent assistant answering questions about Mahitha's resume and personal profile.

Use the extracted context below, **but do not limit yourself to it**.
If something is not directly stated, make logical inferences based on:
- Her experience
- Her listed skills and technologies
- Personal interests (if mentioned)
- Standard industry practices

If the question asks for contact information and it's mentioned in the resume, provide it directly.

If the answer cannot be found or reasonably inferred, respond: 
"This information isn't available in Mahitha’s professional or personal profile."

Be thoughtful and confident.

Resume Context:
{context}

Conversation so far:
{chat_history}

Question: {question}
Answer:
"""
)

# --- Chain Setup ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=False
)

stuff_chain = StuffDocumentsChain(
    llm_chain=LLMChain(llm=llm, prompt=custom_prompt),
    document_variable_name="context"
)


question_generator = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
))

qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    memory=memory,
    combine_docs_chain=stuff_chain,
    question_generator=question_generator,
    return_source_documents=True,
    output_key="answer"
)

# --- Streamlit UI ---
st.set_page_config(page_title="Ask Mahitha (Portfolio QA)", layout="centered", initial_sidebar_state="collapsed")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        div.block-container {padding-top: 0rem;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load and encode background image
with open("mahitha_bg.jpeg", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background: 
            linear-gradient(
                rgba(255, 255, 255, 0.95), 
                rgba(255, 255, 255, 0.95)
            ),
            url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: top center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: blur(0px);
    }}

    /* Apply blur only to the background using a pseudo-element */
        .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: top center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: blur(1px);
        z-index: -999;
        opacity: 0.2;
    }}

    /* Remove the after pseudo-element */
    .stApp::after {{
        display: none;
    }}

    /* Dark theme overlay */
    @media (prefers-color-scheme: dark) {{
        .stApp {{
            background: 
                linear-gradient(
                    rgba(14, 17, 23, 0.95), 
                    rgba(14, 17, 23, 0.95)
            ), 
            url("data:image/jpeg;base64,{encoded}");
        }}

        .stApp::before {{
            opacity: 0.1;
        }}
    }}

    /* Hide ALL empty elements */
    div[data-testid="stMarkdown"]:empty,
    div[data-testid="stMarkdown"] p:empty,
    div[data-testid="stMarkdown"] div:empty,
    div[data-testid="element-container"]:empty,
    div[data-testid="stVerticalBlock"] > div:empty,
    .element-container:empty,
    .stMarkdown:empty {{
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }}

    /* Hide containers that only contain whitespace */
    div[data-testid="stMarkdown"]:not(:has(*)):not(:has(h1)):not(:has(h2)):not(:has(h3)):not(:has(h4)):not(:has(h5)):not(:has(h6)):not(:has(p)):not(:has(ul)):not(:has(ol)):not(:has(li)) {{
        display: none !important;
    }}

    /* Content container styling - only for the main intro container */
    .main-intro-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2.5rem;
        border-radius: 0.8rem;
        margin: -0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }}

    /* Remove background from all other small markdown containers */
    div[data-testid="stMarkdown"]:not(.main-intro-container) {{
        background: transparent !important;
        padding: 0.5rem 0 !important;
        box-shadow: none !important;
        margin: 0.2rem 0 !important;
    }}

    @media (prefers-color-scheme: dark) {{
        .main-intro-container {{
            background-color: rgba(14, 17, 23, 0.85);
        }}
    }}

    /* Headings and text in light mode */
    h1, h2, h3, h4, h5, h6 {{
        color: #1E1E1E !important;
        margin-bottom: 1rem;
    }}

    p, span, label {{
        color: #2D3748 !important;
        line-height: 1.6;
    }}

    /* Headings and text in dark mode */
    @media (prefers-color-scheme: dark) {{
        h1, h2, h3, h4, h5, h6 {{
            color: #FFFFFF !important;
        }}

        p, span, label {{
            color: #E2E8F0 !important;
        }}
    }}

    /* Input field styling */
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1E1E1E !important;
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 0.5rem;
        padding: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}

    @media (prefers-color-scheme: dark) {{
        .stTextInput > div > div > input {{
            background-color: rgba(14, 17, 23, 0.9) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
    }}

    /* Button styling */
    .stButton > button {{
        background-color: #2E86C1 !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .stButton > button:hover {{
        background-color: #2874A6 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }}

    @media (prefers-color-scheme: dark) {{
        .stButton > button {{
            background-color: #3498DB !important;
        }}

        .stButton > button:hover {{
            background-color: #2E86C1 !important;
        }}
    }}

    /* Chat message styling */
    div[data-testid="stMarkdown"] > div > p {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}

    @media (prefers-color-scheme: dark) {{
        div[data-testid="stMarkdown"] > div > p {{
            background-color: rgba(14, 17, 23, 0.9);
        }}
    }}

    /* Fix container width */
    .block-container {{
        max-width: 1000px;
        padding: 2rem 1rem;
    }}

    /* Additional hiding rules for any remaining empty elements */
    div:empty:not([data-testid="stTextInput"]):not([data-testid="stButton"]):not(.stSpinner) {{
        display: none !important;
    }}

    /* Enhanced Question Section Styling */
    .stForm {{
        background: linear-gradient(135deg, rgba(46, 134, 193, 0.95), rgba(52, 152, 219, 0.95)) !important;
        padding: 2rem !important;
        border-radius: 1rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        transform: scale(1.02);
        transition: all 0.3s ease !important;
    }}

    .stForm:hover {{
        transform: scale(1.03) !important;
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.3) !important;
    }}

    /* Question Label Styling */
    .stForm label {{
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 1.5rem !important;
        display: block !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        text-align: center !important;
    }}

    /* Enhanced Input Field Styling */
    .stForm .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.98) !important;
        color: #1E1E1E !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 0.75rem !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }}

    /* Hide the form submission tooltip */
    .stForm .stTextInput [data-testid="InputInstructions"] {{
        display: none !important;
    }}

    .stForm .stTextInput .st-emotion-cache-16txtl3 {{
        display: none !important;
    }}

    /* Hide tooltip specifically without affecting input */
    .stForm [role="tooltip"] {{
        display: none !important;
    }}

    /* Hide help text that appears below input */
    .stForm .stTextInput > div > div:last-child:not(:has(input)) {{
        display: none !important;
    }}

    .stForm .stTextInput > div > div > input:focus {{
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px) !important;
    }}

    /* Enhanced Submit Button Styling */
    .stForm .stButton > button {{
        background: linear-gradient(135deg, #E74C3C, #C0392B) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 6px 15px rgba(231, 76, 60, 0.4) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }}

    .stForm .stButton > button:hover {{
        background: linear-gradient(135deg, #C0392B, #A93226) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(231, 76, 60, 0.6) !important;
    }}

    /* Clear Chat Button Enhancement */
    .stButton:not(.stForm .stButton) > button {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1)) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    }}

    .stButton:not(.stForm .stButton) > button:hover {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2)) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2) !important;
    }}

    /* Chat History Styling - Light Theme Only */
    div[data-testid="stMarkdown"]:has(p:contains("🧑 You")),
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) {{
        padding: 1rem 1.5rem !important;
        border-radius: 0.75rem !important;
        margin: 0.75rem 0 !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }}

    /* Style containers that immediately follow answer containers (for bullet points) - Light Theme */
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"],
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"],
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"] {{
        padding: 1rem 1.5rem !important;
        border-radius: 0.75rem !important;
        margin: 0.25rem 0 !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-top: 2px solid rgba(255, 255, 255, 0.5) !important;
    }}

    /* Alternative approach: Style all markdown containers that contain lists - Light Theme */
    div[data-testid="stMarkdown"]:has(ul):not(.main-intro-container),
    div[data-testid="stMarkdown"]:has(ol):not(.main-intro-container) {{
        padding: 1rem 1.5rem !important;
        border-radius: 0.75rem !important;
        margin: 0.25rem 0 !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }}

    /* Light Theme - White Backgrounds */
    @media (prefers-color-scheme: light) {{
        div[data-testid="stMarkdown"]:has(p:contains("🧑 You")),
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")),
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"]:has(ul):not(.main-intro-container),
        div[data-testid="stMarkdown"]:has(ol):not(.main-intro-container) {{
            background: rgba(255, 255, 255, 0.9) !important;
        }}
    }}

    /* Remove nested styling conflicts */
    div[data-testid="stMarkdown"]:has(p:contains("🧑 You")) p,
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) p,
    div[data-testid="stMarkdown"]:has(p:contains("🧑 You")) ul,
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) ul,
    div[data-testid="stMarkdown"]:has(p:contains("🧑 You")) ol,
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) ol,
    div[data-testid="stMarkdown"]:has(p:contains("🧑 You")) li,
    div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) li {{
        background: transparent !important;
        margin: 0.25rem 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}

    @media (prefers-color-scheme: dark) {{
        .main-intro-container {{
            background-color: rgba(14, 17, 23, 0.85);
        }}

        .stForm .stTextInput > div > div > input {{
            background-color: rgba(14, 17, 23, 0.95) !important;
            color: #FFFFFF !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
        }}

        .stForm .stTextInput > div > div > input:focus {{
            border: 2px solid rgba(52, 152, 219, 0.8) !important;
        }}

        /* Dark Theme - Dark Backgrounds for ALL chat containers */
        div[data-testid="stMarkdown"]:has(p:contains("🧑 You")),
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")),
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"]:has(p:contains("📄 Answer")) + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"] + div[data-testid="stMarkdown"],
        div[data-testid="stMarkdown"]:has(ul):not(.main-intro-container),
        div[data-testid="stMarkdown"]:has(ol):not(.main-intro-container) {{
            background: rgba(14, 17, 23, 0.9) !important;
            border-top: 2px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3) !important;
        }}

        .stApp::before {{
            opacity: 0.1;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class="main-intro-container">
<h3>👋 Hey, I'm Mahitha!</h3>
<p>You're in the right place if you're looking for someone who codes, creates & caffeinates responsibly ☕💻</p>

<p><strong>Curious to know more about me?</strong><br>
Why just read when you can interact? Talk to my chatbot — it's been trained on me, by me, for you 🤖💬</p>

<h4>💡 Try asking things like:</h4>
<ul>
<li><strong>What programming languages does Mahitha know?</strong></li>
<li><strong>What certifications does Mahitha have?</strong></li>
<li><strong>What are Mahitha's hobbies or interests?</strong></li>
<li><strong>What are her career goals?</strong></li>
</ul>

<p>Get creative — if it's about me, my bot probably knows 😉</p>

<p>Go ahead, ask away!</p>
</div>
""", unsafe_allow_html=True)

# --- Input field ---
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input("Ask a question:", placeholder="Type your question here... 💭")
    submitted = st.form_submit_button("Send")

# --- Response processing ---
if submitted and query:
    with st.spinner("Thinking..."):
        if any(keyword in query.lower() for keyword in ["contact", "linkedin", "email", "phone", "reach", "number"]):
            response = (
                "You can reach Mahitha at **(832)-387-5632**, "
                "email: **mahithareddy921@gmail.com**, "
                "or connect on [LinkedIn](https://www.linkedin.com/in/mahithardy/)."
            )
        elif any(phrase in query.lower() for phrase in ["where is she now", "current job", "currently working", "where does she work now"]):
            response = (
        "Mahitha Reddy is currently working at **McKinsey & Co.** in Texas "
                "as a **Software Engineer (Computer Systems Analyst)**."
            )
        else:
            result = qa_chain.invoke({"question": query})
            response = result["answer"]
            if response.strip().lower() in [
                "i don't know.", "i don't have that information.", "not sure.", "i'm not sure."
            ]:
                response = "This information isn't available in Mahitha's professional or personal profile."

        st.session_state.history.insert(0, ("🧑 You", query))
        st.session_state.history.insert(0, ("📄 Answer", response))

if st.button("🗑️ Clear Chat"):
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.markdown(f"**{role}:** {msg}")
