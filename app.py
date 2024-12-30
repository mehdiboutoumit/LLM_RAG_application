from langchain_community.llms import HuggingFaceHub


# Open-source LLM from Hugging Face
llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_length": 256}, huggingfacehub_api_token="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")

# Define the context and the question
context = "You're an assistant knowledgeable about healthcare. Only answer healthcare-related questions. Otherwise say \"I only answer healthcare related question \""
question = "What is LLM?"

# Combine context with the question
input_text = f"{context} Question: {question}"

# Get the response from the model
llm_out = llm.invoke(input_text)

# Print the output
print(llm_out)
