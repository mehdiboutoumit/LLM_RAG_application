from langchain.schema.messages import HumanMessage, SystemMessage
from transformers import pipeline

# Initialize the Hugging Face question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# The context that will be used for answering
context = """
Medicaid managed care is a healthcare delivery system in which states contract with managed care organizations (MCOs) to provide healthcare services to Medicaid beneficiaries.
"""

# Create a conversation where the model understands the task
messages = [
    SystemMessage(
        content="""You're an assistant knowledgeable about
        healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="How to write a medical letter?"),  # Modify this to test with other questions
]

# Define a function to determine relevance based on keywords
def is_relevant(question, context):
    keywords = ["healthcare", "medical", "Medicaid", "care", "beneficiaries", "services"]
    return any(keyword.lower() in question.lower() for keyword in keywords)

# Function to process messages and answer using the QA pipeline
def invoke_qa_pipeline(messages):
    # Retrieve the question from the conversation
    question = None
    for message in messages:
        if isinstance(message, HumanMessage):
            question = message.content
    
    # Check if the question is relevant
    if not is_relevant(question, context):
        return "AIMessage(content='I'm sorry, I can only answer healthcare-related questions.')"
    
    # Perform the QA task using the Hugging Face pipeline
    try:
        result = qa_pipeline(question=question, context=context)
        return f"AIMessage(content='{result['answer']}')"
    except Exception:
        return "AIMessage(content='I couldn’t find an answer to your question within the context provided.')"

# Call the function to simulate the conversation and answer
response = invoke_qa_pipeline(messages)
print(response)
