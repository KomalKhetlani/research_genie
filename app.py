import streamlit as st
from src.retrieval_pipeline import generate_answer

st.set_page_config(page_title="Research Genie", layout="wide")
st.title("Research Genie")
st.markdown(
    "<p style='font-size:16px; color:gray;'>Your AI Assistant for NLP & Large Language Models.</p>",
    unsafe_allow_html=True
)

#Initialise session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history =[]

# Sidebar with "New Conversation" button
with st.sidebar:
    st.header("Char Settings")
    if st.button("🆕 New Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# Display chat messages from session history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("ai"):
        st.write(msg["ai"])

# User input box
query = st.chat_input("Ask me anything...")

if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query, st.session_state.chat_history)

        # Append to chat history
        st.session_state.chat_history.append({"user": query, "ai": answer})

        # Display response
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("ai"):
            st.write(answer)