import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

st.title("🦜🔗 Langchain - Blog Outline Generator App")

st.subheader("I am an experienced data scientist and technical writer. Give me a topic, and I'll generate on outline for a blog post about it",divider='rainbow')
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def blog_outline(topic):
    #Instantiate LLM Model
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
    # Prompt
    template = "As an experienced data scientist and technical writer, generate an outline for a blog post about {topic}."
    
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    #Run LLM Model
    response = llm(prompt_query)
    #Print results
    return st.info(response)

with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        blog_outline(topic_text)