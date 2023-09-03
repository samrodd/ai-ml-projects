import tweepy
import airtable from Airtable
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
import schedule
import time
import os

# helpful when testing locally
from dotenv import load_dotenv
load_dotenv()

# Instantiate LLM
llm = ChatOpenAI(temperature=0.5, open_api_key=OPENAI_API_KEY, model_name='gpt-4')

# Define function to generate Tweet response
def generate_response(llm, mentioned_parent_tweet_text):
    system_template="""
        You are an incredibly wise and smart tech mad scientist from silicon valley.
        Your goal is to give a concise prediction in response to a piece of text from the user.

        % RESPONSE TONE:
        
        Your prediction should be given in an active voice and be opinionated.
        Your tone should be serious w/ a hint of wit and sarcasm.

        % RESPONSE FORMAT:

        - Respond in under 200 characters
        - Respond in two or less short sentences
        - Repond without emojis

        % RESPONSE CONTENT:
        - Include specific examples of old tech if they are relevant
        - If you don't have an answer, say, "Sorry, my magic 8 ball isn't working right now"
    """
    system_message_prompt = SystemMessagePromptTemplate.from_templae(system_template)

    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    final_prompt = chat_prompt.format_prompt(text=mentioned_parent_tweet_text).to_messages()
    response = llm(final_prompt).content

    return response

tweet = """
    Most SaaS founders I've talked to that are AI-first can't explain to me how what they are doing is defensible
    Not looking for a perfect answer, just some sort of real insight
"""

response = generate_response(llm, tweet)
print(response)