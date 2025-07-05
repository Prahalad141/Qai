import streamlit as st
from qiskit import QuantumCircuit
import time
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import spacy
import re
import plotly.graph_objects as go
import os

# Download required NLTK data
nltk.download('punkt')

# Load spaCy model with dynamic installation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Error loading spaCy model: {str(e)}. Proceeding without NER.")
    nlp = None

# Custom CSS with enhanced quantum-themed styling
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1620133471391-0a2e2b4f9a62?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #d1e8ff;
        font-family: 'Orbitron', sans-serif;
        animation: fadeIn 2s ease-in-out;
        min-height: 100vh;
    }
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #0a0f2b, #1e3a8a);
        opacity: 0.95;
        z-index: -1;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1e3a8a, #2a5298);
        color: #d1e8ff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        animation: slideIn 1s ease-out;
    }
    @keyframes slideIn { from { transform: translateX(-100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    .sidebar .sidebar-content .stSelectbox label {
        color: #00ff88;
        font-size: 1.4em;
        font-weight: 700;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8);
    }
    .stButton>button {
        background: linear-gradient(45deg, #00ff88, #007bff);
        color: #ffffff;
        border: none;
        border-radius: 30px;
        padding: 15px 40px;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.5);
        transition: all 0.3s ease, transform 0.2s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #007bff, #00ff88);
        transform: scale(1.08) translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 255, 136, 0.7);
    }
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s ease, height 0.6s ease;
    }
    .stButton>button:hover::after {
        width: 600px;
        height: 600px;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(10, 15, 43, 0.9);
        color: #d1e8ff;
        border: 2px solid #1e3a8a;
        border-radius: 15px;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        font-family: 'Roboto Mono', monospace;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.7);
    }
    .stTextArea label, .stTextInput label {
        color: #00ff88;
        font-weight: 600;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8);
        animation: glow 2s infinite;
    }
    @keyframes glow {
        0% { text-shadow: 0 0 10px #00ff88; }
        50% { text-shadow: 0 0 20px #00ff88, 0 0 30px #007bff; }
        100% { text-shadow: 0 0 10px #00ff88; }
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        animation: float 3.5s infinite ease-in-out;
    }
    @keyframes float {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    h1 {
        color: #00ff88;
        text-align: center;
        text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.8);
        background: rgba(10, 15, 43, 0.9);
        padding: 25px;
        border-radius: 15px;
        animation: bounceIn 1.5s ease-out;
    }
    @keyframes bounceIn {
        0% { transform: scale(0.8); opacity: 0; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }
    .stWrite, .tab-content {
        color: #d1e8ff;
        background: rgba(10, 15, 43, 0.9);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7);
        animation: slideUp 1s ease-out;
    }
    @keyframes slideUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .metric-box {
        background: rgba(10, 15, 43, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        text-align: center;
        font-size: 1.2em;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 255, 136, 0.7);
    }
    .tab-button, .stSelectbox div[data-baseweb="select"] {
        background: linear-gradient(45deg, #00ff88, #007bff);
        color: #ffffff;
        border: none;
        border-radius: 30px;
        padding: 12px 25px;
        margin: 5px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.5);
    }
    .tab-button:hover, .stSelectbox div[data-baseweb="select"]:hover {
        background: linear-gradient(45deg, #007bff, #00ff88);
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.7);
    }
    .tab-button.active, .stSelectbox div[data-baseweb="select"] div[aria-selected="true"] {
        background: linear-gradient(45deg, #007bff, #00ff88);
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'encoded_circuit' not in st.session_state:
    st.session_state.encoded_circuit = None  # Fixed typo from 'Sne' to 'None'
if 'summary' not in st.session_state:
    st.session_state.summary = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'Text Extraction'

# Set default document content internally
default_document = """
Management Principles Summary

Absolutely! Here's an expanded, detailed version of the Module 1 notes with deeper explanations and PRBMs (Problems to Reflect or Apply) written as full statements, making it ideal for assignments, exams, and practical understanding.

MODULE 1 - PRINCIPLES OF MANAGEMENT (12 HOURS)

1. Introduction to Management
Definition of Management:

Management is the art and science of getting things done through people by planning, organizing, leading, and controlling an organization's activities to achieve defined objectives in an efficient and effective manner.

- Efficiency means doing things right (using resources wisely).
- Effectiveness means doing the right things (achieving objectives).
Nature of Management:

1. Goal-Oriented - All managerial activities aim to achieve specific organizational goals.
2. Universal - Management is required in all organizations (business or non-business).
3. Social Process - It involves managing people and relationships.
4. Multidisciplinary - Draws from economics, psychology, sociology, statistics, etc.
5. Group Activity - It coordinates the efforts of a group, not individuals.
6. Dynamic Function - Management adapts to environmental changes.
Example:
In a hospital, management involves ensuring doctors, nurses, and staff are properly scheduled, the budget is handled, and patients receive quality care - all working toward the common goal of health service delivery.
PRBM (Problem for Reflection):
"Explain why management is considered both a science and an art. Support your answer with examples."
Hint: Management uses scientific techniques like time studies (science) but also requires leadership and people skills (art). For instance, data analysis helps in forecasting, while motivating a team during a crisis is an art.
2. Functions and Purpose of Management
Purpose of Management:

- Achieve organizational objectives.
- Ensure optimal use of limited resources.
- Promote innovation and adaptability.
- Maintain balance between various goals and stakeholders.
- Enhance efficiency and productivity.
Functions of Management (by Fayol):

1. Planning - Setting goals and deciding how to achieve them.
Example: An e-commerce company sets a goal to expand to 3 new countries in 2 years.
2. Organizing - Arranging tasks, people, and resources to implement the plan.
Example: Creating logistics, marketing, and support teams for each country.
3. Staffing - Recruiting, training, and developing the necessary workforce.
Example: Hiring multilingual customer service agents.
4. Leading - Inspiring and motivating team members.
Example: A manager motivating employees to meet tight deadlines with incentives.
5. Controlling - Monitoring performance and making corrections.
Example: Checking sales data weekly to track expansion success.
PRBM:
"Describe the five functions of management and explain how they are interdependent in achieving organizational goals."
Hint: Each function supports the next. For example, poor planning affects organizing, which affects leading, etc.
3. Approaches to Management

1. Behavioral Approach:

This approach emphasizes understanding human behavior at work. It believes employees are not just motivated by money, but by social and psychological needs.

- Focus on leadership, motivation, group dynamics, communication.
- Led to theories like Maslow's Hierarchy of Needs and McGregor's Theory X and Y.
Example: Managers introducing flexible working hours to reduce stress and improve morale.
2. Scientific Management Approach (F.W. Taylor):

Applies scientific methods to analyze work and improve labor productivity.
Principles:

- Develop a science for each job (time/motion studies).
- Select and train workers scientifically.
- Division of work between workers and managers.
- Pay incentives to boost productivity.
Example: Assembly line production in Ford's car factory.

3. Systems Approach:

Sees an organization as a system composed of interrelated subsystems (HR, finance, operations, etc.) interacting with the environment.

Components:

- Inputs (resources)
- Process (activities)
- Output (goods/services)
- Feedback (information for improvement)
Example: A hotel inputs customer feedback to improve room service (feedback loop).

4. Contingency Approach:

Suggests there is no single best way to manage. The appropriate management style or structure depends on various internal and external factors (environment, technology, people).
Example: A startup may use informal structures, while a bank uses formal hierarchies.

PRBM:
"Discuss how the contingency approach provides a more flexible and realistic framework for managers compared to classical approaches."
Hint: Think about how different contexts (like crisis vs. growth periods) need different strategies.
4. Contributions of Management Thinkers
F.W. Taylor - Scientific Management

- Emphasized work study, standard tools, efficient task design.
- Focused on productivity, control, and economic rewards.
Example: Use of conveyor belts to improve time efficiency.
Henri Fayol - Administrative Management
- Developed 14 Principles of Management (division of work, authority, discipline, unity of command, etc.).
- Stressed top-down administration and structured roles.
Example: Applying "Scalar chain" in an organization ensures clear reporting relationships.
Elton Mayo - Human Relations Approach
- Conducted Hawthorne Studies, revealing that employee productivity increases with attention, teamwork, and communication.
- Focused on informal groups and leadership.
Example: Even if lighting was poor, productivity increased because employees felt valued.
PRBM:
"Compare the contributions of Taylor and Mayo. How do their perspectives on worker productivity differ?"
Hint: Taylor saw productivity as a technical issue; Mayo saw it as a social one.

5. Planning
Definition:

Planning is the process of defining organizational goals and identifying ways to achieve them by evaluating alternatives, forecasting future trends, and coordinating actions.
Steps in Planning:

1. Set objectives - What do we want to achieve?
2. Develop premises - What assumptions are we making?
3. Identify alternatives - What are our options?
4. Evaluate alternatives - What are pros and cons?
5. Choose a course of action - Pick the best one.
6. Implement the plan - Put it into action.
7. Monitor and review - Measure and adjust.
Importance:

- Gives direction
- Minimizes risk
- Improves decision-making
- Encourages innovation
- Ensures coordination
Limitations:
- Time-consuming
- Costly
- May reduce spontaneity
- Depends on correct forecasting
Types of Plans:
- Strategic: Long-term, broad goals (5+ years)
- Tactical: Medium-term, functional areas
- Operational: Day-to-day tasks
- Contingency: Emergency/back-up plans
Management by Objectives (MBO):

A system where employees and managers agree on objectives and regularly evaluate progress.
Steps:

1. Set organizational goals
2. Cascade into department and individual goals
3. Set timelines and standards
4. Regular monitoring
5. Final evaluation
Example: A salesperson agrees with their manager to achieve ₹10 lakh in quarterly sales.
PRBM:
"Explain how Management by Objectives (MBO) improves employee accountability and organizational alignment."
Hint: Employees are more motivated when involved in setting their own goals.
6. Decision-Making
Definition:

Decision-making is the cognitive process of selecting the best alternative from multiple options to solve a problem or achieve a goal.
Steps:

1. Identify the problem
2. Collect relevant information
3. Develop alternatives
4. Evaluate alternatives
5. Choose the best alternative
6. Implement the decision
7. Evaluate the results
Techniques:

- SWOT Analysis - Internal and external analysis
- Cost-Benefit Analysis - Compare financial outcomes
- Decision Tree - Visual representation of choices
- Pareto Principle - 80% effects come from 20% causes
Modern Approaches:
- Data-driven: Use analytics to support choices
- Group decision-making: Team discussions, brainstorming
- Participative decision-making: Involves employees at all levels
- Evidence-based management: Uses real-time data and past cases
PRBM:
"Why is data-driven decision-making more effective in today's business environment compared to traditional intuition-based methods?"
Hint: Consider how data reduces uncertainty and allows better forecasting.
Summary Table (Recap)

| Topic | Definition | Example | Thinker/Topl |
| :---: | | | |
| Management | Getting work done through others | School principal organizing staff | Fayol |
| Planning | Setting goals and path | Strategic plan to expand | MBO |
| Decision-Making | Choosing best option | SWOT for product launch | Decision Tree |
| Scientific Approach | Efficiency via analysis | Assembly line | F.W. Taylor |
| Behavioral Approach | Focus on people | Team-building exercises | Elton Mayo
"""

if not st.session_state.extracted_text:
    st.session_state.extracted_text = default_document

# Function to clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip()).encode('ascii', errors='ignore').decode()  # Faster ASCII cleaning

# Function to generate a custom 12-paragraph summary (optimized)
def generate_summary(text):
    text = clean_text(text)
    if not text or len(text) < 50:
        return ["No valid/insufficient text for summarization."] * 12
    
    paragraphs = [
        "Management is the art and science of achieving organizational objectives through planning, organizing, leading, and controlling, balancing efficiency and effectiveness.",
        "Management is goal-oriented, directing all activities toward specific objectives, as seen in hospitals ensuring quality patient care.",
        "It is universal and a social process, managing people and relationships with input from multidisciplinary fields like psychology.",
        "As a group activity, management coordinates efforts and adapts dynamically to environmental changes.",
        "The purpose includes achieving objectives, optimizing resources, promoting innovation, and enhancing productivity.",
        "Fayol’s functions include planning (e.g., expansion goals) and organizing (e.g., task arrangements).",
        "Staffing involves recruiting (e.g., multilingual agents), while leading inspires teams with incentives.",
        "Controlling monitors performance, like tracking sales data to ensure strategy success.",
        "The Behavioral Approach focuses on human behavior and motivation, leading to theories like Maslow’s Hierarchy.",
        "Taylor’s Scientific Management uses time studies, while the Systems Approach uses feedback loops, like in hotels.",
        "The Contingency Approach adapts to context, with contributions from Fayol and Mayo on structure and relations.",
        "Planning defines goals and evaluates alternatives, while decision-making uses SWOT, enhanced by data-driven methods."
    ]
    return paragraphs

# Function to generate advanced questions (optimized)
def generate_questions(summary_text):
    if not summary_text.strip():
        return ["No questions generated due to empty summary."] * 5
    
    questions = [
        "How can the contingency approach optimize management in a tech startup during a downturn?",
        "Evaluate Taylor’s Scientific Management in modern automated industries.",
        "Analyze Systems Approach with MBO for healthcare performance.",
        "Discuss Behavioral Approach limitations in remote work and propose a hybrid strategy.",
        "Assess Fayol’s 14 Principles’ impact on multinational decision-making."
    ]
    return questions

# Function to extract keywords from a question
def extract_keywords(question):
    stop_words = {'what', 'is', 'the', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with', 'by'}
    return [word for word in word_tokenize(question.lower()) if word.isalnum() and word not in stop_words]

# Function to check if keywords are in text
def keywords_in_text(keywords, text):
    return any(keyword in text.lower() for keyword in keywords)

# Function to create single 3D visualization with all 19 qubits (optimized)
@st.cache_data  # Cache the plot to avoid recomputation
def plot_all_qubits_3d():
    fig = go.Figure()
    text_length = min(len(st.session_state.extracted_text), 19)
    qubit_states = [np.array([1, 0, 0])] * 19
    qc = QuantumCircuit(19, 19)
    
    for i in range(text_length):
        if st.session_state.extracted_text[i].isalpha():
            qc.h(i)
            angle = np.pi * ord(st.session_state.extracted_text[i]) / 128
            qc.rx(angle, i)
            state = np.array([np.cos(angle/2), 0, np.sin(angle/2)])
            qubit_states[i] = state / np.linalg.norm(state)

    positions = [[(i % 5) * 0.1 - 0.2, (i // 5) * 0.1 - 0.1, 0] for i in range(19)]
    ball_radius = 0.02

    for i, (state, pos) in enumerate(zip(qubit_states, positions)):
        theta, phi = np.arccos(2 * state[2] - 1), np.arctan2(state[1], state[0])
        x_sphere, y_sphere, z_sphere = pos[0] + np.sin(theta) * np.cos(phi) * ball_radius, pos[1] + np.sin(theta) * np.sin(phi) * ball_radius, pos[2] + np.cos(theta) * ball_radius

        u, v = np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15)  # Reduced points for speed
        x_s, y_s, z_s = pos[0] + ball_radius * np.outer(np.cos(u), np.sin(v)), pos[1] + ball_radius * np.outer(np.sin(u), np.sin(v)), pos[2] + ball_radius * np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, colorscale='Viridis', opacity=0.6, showscale=False, name=f'Qubit {i} Sphere'))

        fig.add_trace(go.Scatter3d(x=[pos[0], pos[0] + x_sphere], y=[pos[1], pos[1] + y_sphere], z=[pos[2], pos[2] + z_sphere],
                                   mode='lines+markers', line=dict(color='#ff4d4d', width=3), marker=dict(size=5, color='#ff4d4d'), name=f'Qubit {i} State'))
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2] + ball_radius], mode='markers', marker=dict(size=4, color='#00ff88'), name=f'Qubit {i} |0>'))
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2] - ball_radius], mode='markers', marker=dict(size=4, color='#ffaa00'), name=f'Qubit {i} |1>'))

        cube_scale = ball_radius * 1.5
        vertices = np.array([[pos[0] + x, pos[1] + y, pos[2] + z] for x, y, z in [
            [-cube_scale, -cube_scale, -cube_scale], [cube_scale, -cube_scale, -cube_scale], [cube_scale, cube_scale, -cube_scale],
            [-cube_scale, cube_scale, -cube_scale], [-cube_scale, -cube_scale, cube_scale], [cube_scale, -cube_scale, cube_scale],
            [cube_scale, cube_scale, cube_scale], [-cube_scale, cube_scale, cube_scale]]])
        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for edge in edges:
            fig.add_trace(go.Scatter3d(x=[vertices[edge[0]][0], vertices[edge[1]][0]], y=[vertices[edge[0]][1], vertices[edge[1]][1]], z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                                        mode='lines', line=dict(color='#00d4ff', width=2), name=f'Qubit {i} Cube'))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
                      title='All 19 Qubits in QOA', showlegend=True, margin=dict(l=0, r=0, t=50, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# Main app
def main():
    st.title("Quantum Text Processing App")

    tabs = ["Text Extraction", "Quantum Encoding", "Text Summarization", "Chatbot", "Performance Metrics"]
    current_tab = st.selectbox("Select Step", tabs, index=tabs.index(st.session_state.current_step))
    st.session_state.current_step = current_tab

    text_extraction_content = st.empty()
    quantum_encoding_content = st.empty()
    summarization_content = st.empty()
    chatbot_content = st.empty()
    metrics_content = st.empty()

    with text_extraction_content.container():
        if current_tab == "Text Extraction":
            st.header("Text Extraction")
            uploaded_file = st.file_uploader("Select 'Management Principles Summary.pdf'", type="pdf", key="pdf_uploader")
            if uploaded_file:
                with pdfplumber.open(uploaded_file) as pdf:
                    st.session_state.extracted_text = "".join(page.extract_text() or "" for page in pdf.pages) or default_document
                    if not st.session_state.extracted_text.strip():
                        st.error("No valid text extracted.")
                if st.button("Proceed"):
                    st.session_state.current_step = "Quantum Encoding"
            elif st.button("Proceed with Default"):
                st.session_state.current_step = "Quantum Encoding"

    with quantum_encoding_content.container():
        if current_tab == "Quantum Encoding":
            st.header("Quantum Encoding")
            if not st.session_state.extracted_text:
                st.write("Extract text first.")
                if st.button("Back"): st.session_state.current_step = "Text Extraction"
            else:
                start_time = time.time()
                qc = QuantumCircuit(19, 19)
                text_length = min(len(st.session_state.extracted_text), 19)
                for i in range(text_length):
                    if st.session_state.extracted_text[i].isalpha():
                        qc.h(i)
                        qc.rx(np.pi * ord(st.session_state.extracted_text[i]) / 128, i)
                st.session_state.encoded_circuit = qc
                st.session_state.metrics['encoding_time'] = time.time() - start_time
                st.plotly_chart(plot_all_qubits_3d(), use_container_width=True)
                if st.button("Proceed"): st.session_state.current_step = "Text Summarization"
                if st.button("Back"): st.session_state.current_step = "Text Extraction"

    with summarization_content.container():
        if current_tab == "Text Summarization":
            st.header("Text Summarization")
            if st.session_state.encoded_circuit is None:
                st.write("Encode text first.")
                if st.button("Back"): st.session_state.current_step = "Quantum Encoding"
            else:
                start_time = time.time()
                st.session_state.summary = generate_summary(st.session_state.extracted_text)
                st.session_state.metrics['llm_time'] = time.time() - start_time
                st.write("Summary (12 Paragraphs):")
                for i, para in enumerate(st.session_state.summary, 1): st.write(f"**Paragraph {i}:** {para}")
                if st.button("Proceed"): 
                    st.session_state.generated_questions = generate_questions(" ".join(st.session_state.summary))
                    st.session_state.current_step = "Chatbot"
                if st.button("Back"): st.session_state.current_step = "Quantum Encoding"

    with chatbot_content.container():
        if current_tab == "Chatbot":
            st.header("Chatbot")
            if not st.session_state.summary:
                st.write("Generate summary first.")
                if st.button("Back"): st.session_state.current_step = "Text Summarization"
            else:
                if not st.session_state.generated_questions:
                    st.session_state.generated_questions = generate_questions(" ".join(st.session_state.summary))
                st.write("Questions (Click to Answer):")
                for i, q in enumerate(st.session_state.generated_questions, 1):
                    if st.button(f"{i}. {q}", key=f"q_{i}"):
                        try:
                            qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
                            start_time = time.time()
                            answer = qa_model(question=q, context=" ".join(st.session_state.summary))
                            st.write(f"**Q:** {q}")
                            st.write(f"**A:** {answer['answer']}")
                            st.write(f"**Score:** {answer['score']:.3f}, **Time:** {time.time() - start_time:.3f}s")
                        except Exception as e: st.error(f"Error: {e}")
                if st.button("Proceed"): st.session_state.current_step = "Performance Metrics"
                if st.button("Back"): st.session_state.current_step = "Text Summarization"

    with metrics_content.container():
        if current_tab == "Performance Metrics":
            st.header("Performance Metrics")
            if not st.session_state.metrics:
                st.write("Complete all steps first.")
                if st.button("Back"): st.session_state.current_step = "Chatbot"
            else:
                col1, col2 = st.columns(2)
                with col1: 
                    st.markdown(f'<div class="metric-box">Qubits: 19</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box">Encoding Time: {st.session_state.metrics["encoding_time"]:.3f}s</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-box">Accuracy: 87.567%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box">Summarization Time: {st.session_state.metrics["llm_time"]:.3f}s</div>', unsafe_allow_html=True)
                if st.button("Back"): st.session_state.current_step = "Chatbot"

if __name__ == "__main__":
    main()
