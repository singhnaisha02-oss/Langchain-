import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini import kiya

# Streamlit page config
st.set_page_config(
    page_title="My Mobile RAG App", page_icon="🤖", layout="centered"
)

st.title("🤖 My RAG App (Powered by Gemini)")
st.write("Google AI Studio ki free API ke sath!")

# Streamlit secrets se Gemini API key uthana
try:
  os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
  st.warning(
      "Please set your GOOGLE_API_KEY in Streamlit Secrets dashboard settings."
  )

# Sample Document Context
context_text = """
Naisha Engineers Private Limited handles high-stakes Pan-India operations. 
We specialize in industrial safety systems and digital mining logs portals.
"""

# User Input
user_query = st.text_input("Apna sawaal pucho (e.g., What does Naisha Engineers do?):")

if user_query:
  with st.spinner("Thinking..."):
    # Prompt & Gemini LLM Setup
    template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)

    # Gemini ka lightweight aur fast model use kar rahe hain
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key="AQ.Ab8RN6J4Xt56mSeHXqqotU3rkR29M08iCTurRCfADDAmMAZBNA",

    # ya fir st.secrets["GOOGLE_API_KEY"]
    vertexai=False,  # Yeh line sabse important hai!
)



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
