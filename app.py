import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Streamlit page config
st.set_page_config(
    page_title="My Mobile RAG App", page_icon="🤖", layout="centered"
)

st.title("🤖 My RAG App (Powered by Streamlit)")
st.write("Phone se code karo, cloud pe run karo!")

# Streamlit secrets se OpenAI API key uthana (Security ke liye best hai)
try:
  os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
  st.warning(
      "Please set your OPENAI_API_KEY in Streamlit Secrets dashboard settings."
  )

# Sample Document Context (Ise baad mein vector DB ya file uploader se connect kar sakti ho)
context_text = """
Naisha Engineers Private Limited handles high-stakes Pan-India operations. 
We specialize in industrial safety systems and digital mining logs portals.
"""

# User Input
user_query = st.text_input("Apna sawaal pucho (e.g., What does Naisha Engineers do?):")

if user_query:
  with st.spinner("Thinking..."):
    # Prompt & LLM Setup
    template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Chain
    rag_chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_query)

    st.success("Done!")
    st.write("### Answer:")
    st.write(response)
