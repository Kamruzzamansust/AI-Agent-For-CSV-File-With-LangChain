import streamlit as st
import pandas as pd 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os 
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
CSV_PROMPT_PREFIX = """ 
First set the pandas display options to show all the columns 
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """ 
- **ALWAYS** before giving the final answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself: Which one is more accurate?
If you are not sure, try another method.

Format 4 FIGURES OR MORE WITH COMMAS.

- If the methods tried do not give the same result, reflect, 
and try again until you have two methods that have the same result.
- If you still cannot arrive at a correct result, say that you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful and thorough response using markdown.
"""

st.title("AI Agent For CSV File With LangChain")

uploadedFile = st.file_uploader("Upload Your File", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")

if uploadedFile:
    df = pd.read_excel(uploadedFile)  # Read the uploaded Excel file
    st.write(df.head())  # Display the first few rows of the dataframe
    
    agent = create_pandas_dataframe_agent(
        llm=ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It", temperature=0.7),
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True  # Enable parsing error handling
    )

    st.write('### Ask A Question')
    question = st.text_input(
        "Enter your question about the dataset:",
        "What is the total number of unique subproducts under the Products?"
    )

    if st.button("Run Query"):
        QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
        res = agent.invoke(QUERY)
        st.write(res['output'])
