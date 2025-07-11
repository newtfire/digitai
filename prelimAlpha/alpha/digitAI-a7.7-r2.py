from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from llama_index.readers.file import XMLReader
import xml.etree.ElementTree as ET
from lxml import etree

llm = OllamaLLM(model="llama3.2", temperature=0.5)

log_file = "../output/chat_history18.txt"
dataset_file = "../p5subset.xml"

# Load XML data using LlamaIndex's XMLReader
def load_xml_data(file_path):
  try:
    reader = XMLReader()
    document = reader.load_data(file_path) # Load XML as structured document
    return document[0].text # Extract text content
  except Exception as e:
    return f"Error loading XML with LlamaIndex: {e}"
# Load dataset from XML file
dataset_content = load_xml_data(dataset_file)

chat_history = [dataset_content]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are an AI.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt_template | llm

tree = etree.parse(dataset_file)
root = tree.getroot()

def start_app():
    with open(log_file, "a") as file:
        file.write(f"\n--- Chat started on {datetime.datetime.now()} ---\n")
    while True:
        question = input("You: ")
        if question.lower() == "done":
            with open(log_file, "a") as file:
                file.write("\n--- Chat Ended ---\n")
            print("Chat history saved. Exiting...")
            return
        else:
            response = chain.invoke({"input": question, "chat_history": chat_history})
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=response))
            print("AI:", response)
            with open(log_file, "a") as file:
                file.write(f"You: {question}\n")
                file.write(f"AI: {response}\n")
if __name__ == "__main__":
    start_app()