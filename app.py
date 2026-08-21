import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq  # Groq import kiya

st.set_page_config(
    page_title="My Mobile RAG App", page_icon="🤖", layout="centered"
)
st.title("🤖 My RAG App (Powered by Groq)")
st.write("Free Llama model ke sath!")

# Yahan apni gsk_ se shuru hone wali Groq key daal do
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    groq_api_key="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",
)

context_text = """
Naisha Engineers Private Limited handles high-stakes Pan-India operations. 
We specialize in industrial safety systems and digital mining logs portals.
"""

user_query = st.text_input("Apna sawaal pucho:")

if user_query:
  with st.spinner("Thinking..."):
    template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)

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
