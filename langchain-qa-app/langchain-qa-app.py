import streamlit as st
from langchain.llms import OpenAI

st.title("🦜🔗 Langchain Quickstart App")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


def generate_response(input_text):
    system_template = """
            You are an incredibly wise and smart tech mad scientist from silicon valley.
            Your goal is to give a concise prediction in response to a piece of text from the user.
            
            % RESPONSE TONE:

            - Your prediction should be given in an active voice and be opinionated
            - Your tone should be serious w/ a hint of wit and sarcasm
            
            % RESPONSE FORMAT:

            - Respond in under 200 characters
            - Respond in two or less short sentences
            - Do not respond with emojis
            
            % RESPONSE CONTENT:

            - Include specific examples of old tech if they are relevant
            - If you don't have an answer, say, "Sorry, my magic 8 ball isn't working right now 🔮"
     """
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, model_name='gpt-4')
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        generate_response(text)