import os
import json
import pandas as pd
import traceback
from dotenv import  load_dotenv
from src.logger import logging
from src.utils import get_table_data, read_file
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")



RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}


TEMPLATE_1 = """
You are an expert in MCQ generation. Please generate {number} multiple choice questions about {subject} subject from the delimited text by triple double quotes and question should in {tone} tone. \
Don't create the multiple choices value with same as the previous question multiple choices instead replace with values that are similar to the question\

Text to generate questions:\
{text}

Please make sure that you don't forget about the question tone, it is really important to assess the student knowledge and exclude the delimiter.  If there is nothing related to {subject} subject then please show that \
\"There is no text related to {subject}\".\

Outputs should be in JSON  object following:
{response_json}

"""



TEMPLATE_2 = """
You are experienced professor. You need to evaluate the complexity of multiple choice question delimited by triple double quotes and give a complete analysis of the quiz about {subject}.\
Only use maximum 50 words for complexity analysis. If the quiz is not suitable with the cognitive and analytical abilities of the {subject} students, update the quiz questions \
which needs to be changed and change the tone of the multiple choice questions such that perfectly aligns and fits the {subject} student abilities.Make sure that don't repeat the same \
multiple choice values. Outputs should in JSON object \
that includes the following:\
keys: question, multiple choices, and correct answer

Quiz multiple choice question:
\"\"\"{quiz}\"\"\"

Check from an {subject} expert that the question and answer is valid and correct. Remember don't repeat the same multiple choice values except the answer.
"""

# Creates a new instance of OpenAI
llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.6)

# Create Prompt Template to the quesitons in quiz
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE_1
)

# Create chain to generate the quesitons in quiz
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)


# Create Prompt Template to evaluate the quesitons in quiz
quiz_evaluation_prompt = PromptTemplate(
    input_variables=["quiz", "subject"],
    template = TEMPLATE_2
)

# Create chain to evaluate the quesitons in quiz
quiz_evaluation_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


# Setting up SequentialChain for generating and evaluating the quesitons in quiz
complete_quiz_chain = SequentialChain(
    chains=[quiz_chain, quiz_evaluation_chain], 
    input_variables=["text", "number", "subject", "tone", "response_json"], 
    output_variables=["quiz", "review"],
    verbose=True
 )


# Defining the variables
file_path = 'D:\Machine learning\Generative AI\MCQ generator\data.txt'
TEXT = read_file(file_path)
NUMBER = 10
SUBJECT = 'Physics'
TONE = 'Normal'



# Setup token usage tracking in langchain
with get_openai_callback() as cb:
    response = complete_quiz_chain(
        {
            "text": TEXT,
            "number": NUMBER,
            "subject": SUBJECT,
            "tone": TONE,
            "response_json": json.dumps(RESPONSE_JSON)
        }
    )


# Creating a DataFrame and saving it in csv format
df = get_table_data(response)
df.to_csv('Questions.csv', index=False)

# Printing token usage in langchain
logging.info(cb)


