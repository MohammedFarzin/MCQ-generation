import os
import json
import traceback
import pandas as pd
from dotenv import  load_dotenv
from src.utils import get_table_data, read_file
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.logger import logging
from src.mcq_generator.mcq import generate_mcq_response


# loading json file
with open("response.json", "r") as f:
    RESPONSE_JSON = json.load(f)

# Create a title for the app

st.title("MCQ Generator")
st.markdown("""
This app generates a MCQ based on the given input.
""")


# Create a form using st.form
with st.form("user_inputs"):
    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
    
    # Include input fields
    mcq_count = st.number_input("Number of MCQs", min_value=1, max_value=100)

    # Subject field
    subject = st.text_input("Subject")
    
    # Tone field
    tone = st.selectbox("Select Tone", ["Easy","Normal","Hard"])

    # Submit button
    button = st.form_submit_button("Create")

    # Check if the form was submitted
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)

                # Calculating the total number of tokens and cost
                with get_openai_callback() as cb:
                    responses = generate_mcq_response(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
            
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Something went wrong. Please try again.")
            
            else:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                if isinstance(responses, dict):
                    # Extract the quiz data from the response
                    quiz = responses.get("quiz", None)
                    if quiz:
                        table_data = get_table_data(quiz)
                        if table_data:
                            df = pd.DataFrame(table_data)
                            df.index += 1
                            st.table(df)
                            st.success("Successfully generated the MCQ")
                            st.text_area(label="Review", value=responses.get("review", None), height=200)
                        else:
                            st.error("No data found")
                else:
                    st.write(response)
