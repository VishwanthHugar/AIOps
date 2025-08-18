import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Interactive Chatbot", page_icon="🤖")
st.title("Chatbot (Ollama() + LangChain + memory)")

# ---------------- MODEL SELECTION ----------------
st.sidebar.subheader("Choose Ollama Model")
model_choice = st.sidebar.selectbox(
    "Select a model:",
    ["llama3.2:1b", "gemma3:1b"]  # add any available models here
)
# ---------------- FILE INPUT ----------------
st.sidebar.header("Kpi data Input")

file_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Enter path"])

kpi_data  = None
if file_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload KPI CSV", type="csv")
    if uploaded_file is not None:
        kpi_data  = pd.read_csv(uploaded_file)
elif file_source == "Enter path":
    csv_path = st.sidebar.text_input("CSV file path", value="../result/kpis_with_time_series_anomalies.csv")
    if csv_path:
        try:
            kpi_data  = pd.read_csv(csv_path)
        except Exception as e:
            st.sidebar.error(f" Could not read file: {e}")

if kpi_data  is not None:
    st.write(f"Loaded {len(kpi_data )} rows from KPI dataset.")
else:
    st.warning("Please upload or enter a valid CSV file to continue.")
    st.stop()

# ---- Events File ----
st.sidebar.subheader("Event Data")
event_source = st.sidebar.radio("Choose Event data source:", ["Upload CSV", "Enter path"], key="event")
event_data = None
if event_source == "Upload CSV":
    uploaded_event = st.sidebar.file_uploader("Upload Event CSV", type="csv", key="event_upload")
    if uploaded_event is not None:
        event_data = pd.read_csv(uploaded_event)
elif event_source == "Enter path":
    event_path = st.sidebar.text_input("Event CSV file path", value="", key="event_path")
    if event_path:
        try:
            event_data = pd.read_csv(event_path)
        except Exception as e:
            st.sidebar.error(f" Could not read Event file: {e}")


# ---- Log File ----
st.sidebar.subheader("Log Data")
log_source = st.sidebar.radio("Choose Log data source:", ["Upload CSV", "Enter path"], key="log")
log_data = None
if log_source == "Upload CSV":
    uploaded_log = st.sidebar.file_uploader("Upload log CSV", type="csv", key="log_upload")
    if uploaded_log is not None:
        log_data = pd.read_csv(uploaded_log)
elif log_source == "Enter path":
    log_path = st.sidebar.text_input("Event CSV file path", value="", key="log_path")
    if log_path:
        try:
            log_data = pd.read_csv(log_path)
        except Exception as e:
            st.sidebar.error(f" Could not read Event file: {e}")
# ---------------- INIT LLM ----------------
if "conversation" not in st.session_state:
    # 1. Connect to Ollama model (make sure Ollama is running locally)
    llm = ChatOllama(model="gemma3:1b")  # # pick your model
    
    # 2. Add memory (stores chat history)
    memory = ConversationBufferMemory()

    # 3. Create a prompt template for conversational tone
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template="""
    You are a helpful chatbot. Use the conversation history and reply to the user.

    Conversation history:
    {history}

    User: {input}
    Assistant:"""
    )
    # 4. Build the chain
    st.session_state.conversation = ConversationChain(llm=llm, memory=memory, verbose=True, prompt=prompt_template)
    #st.session_state.conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# ---------------- DISPLAY CHAT ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["message"])
    else:
        st.chat_message("assistant").write(chat["message"])

# ---------------- INPUT ----------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to session history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    st.chat_message("user").write(user_input)

    # Prepare context: take last 20 rows of each file
    context = ""
    if kpi_data is not None:
        context += f"\n--- KPI Data (last 20 rows) ---\n{kpi_data .tail(70).to_string(index=False)}"
    if event_data is not None:
        context += f"\n--- Event Data (last 20 rows) ---\n{event_data.tail(70).to_string(index=False)}"    
    if log_data is not None:
        context += f"\n--- log Data (last 20 rows) ---\n{log_data.tail(70).to_string(index=False)}"

    full_prompt = f"Contex: \n{context}\n\nQuestion: {user_input}"

    # Get LLM response
    response = st.session_state.conversation.predict(input=full_prompt)

    # Add bot response
    st.session_state.chat_history.append({"role": "assistant", "message": response})
    st.chat_message("assistant").write(response)
