from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.1")

template = """
you are a helpful assistant that can answer questions about the following text:
{text}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Ask your question (q to quite):")
    print("\n\n")
    if question == "q":
        break
    result = chain.invoke({"text": question})
    print(result)