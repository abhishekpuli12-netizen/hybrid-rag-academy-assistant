import requests
import streamlit as st

API_URL = "http://backend:8000/chat"

st.set_page_config(
    page_title="Kalashala AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Kalashala AI Assistant")

st.markdown(
    "Ask questions about courses, trainers, batches, fees, locations and admissions."
)

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display History
# ----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):

            with st.expander("📚 Sources"):

                for src in msg["sources"]:

                    st.markdown(
                        f"**{src['file']}**"
                    )

                    st.caption(
                        src["breadcrumb"]
                    )

# ----------------------------
# Chat Input
# ----------------------------

question = st.chat_input(
    "Ask your question..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            try:

                response = requests.post(

                    API_URL,

                    json={
                        "question": question
                    },

                    timeout=300
                )

                result = response.json()

                answer = result["answer"]

                sources = result["sources"]

            except Exception as e:

                answer = f"Error:\n\n{e}"

                sources = []

            st.markdown(answer)

            if sources:

                with st.expander("📚 Sources"):

                    for src in sources:

                        st.markdown(
                            f"**{src['file']}**"
                        )

                        st.caption(
                            src["breadcrumb"]
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )