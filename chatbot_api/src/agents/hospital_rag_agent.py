import os
from chains.hospital_cypher_chain import hospital_cypher_chain
from chains.hospital_review_chain import reviews_vector_chain
from transformers import pipeline
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

# Initialize Hugging Face pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define environment variable for model
HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

# Tools using the Hugging Face pipeline
tools = [
    {
        "name": "Experiences",
        "func": reviews_vector_chain.invoke,
        "description": """Useful when you need to answer questions
        about patient experiences or qualitative analysis using semantic
        search. Not suitable for numerical or statistical data.""",
    },
    {
        "name": "Graph",
        "func": hospital_cypher_chain.invoke,
        "description": """Useful for answering questions about patient statistics,
        hospital visits, or related data. Provide the full prompt as input.""",
    },
    {
        "name": "Waits",
        "func": get_current_wait_times,
        "description": """For current wait times at a hospital. Input should
        be the hospital name, excluding the word "hospital".""",
    },
    {
        "name": "Availability",
        "func": get_most_available_hospital,
        "description": """Find which hospital has the shortest wait time.
        This returns a dictionary with the hospital name and wait time.""",
    },
]

# Hugging Face-based QA invocation
def invoke_qa_tool(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error in QA pipeline: {str(e)}"

# Define a lightweight agent to manage tool selection
def hospital_agent(question, context):
    for tool in tools:
        if tool["name"] == "Graph" and "statistics" in question.lower():
            return tool["func"](question)
        elif tool["name"] == "Waits" and "wait time" in question.lower():
            return tool["func"](question)
        elif tool["name"] == "Availability" and "shortest wait" in question.lower():
            return tool["func"](question)
        elif tool["name"] == "Experiences":
            return invoke_qa_tool(question, context)
    return "No suitable tool found for the query."

# Example usage
context = "Medicaid managed care involves healthcare services provided to beneficiaries."
response = hospital_agent("What is the current wait time at City Hospital?", context)
print(response)
