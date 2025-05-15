import json
import os
import sys

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ---------------------------
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler
from langfuse import Langfuse
import uuid



# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize Langfuse
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

session_name = f"session-{uuid.uuid4().hex[:8]}"
trace_name = "ai-response"
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    trace_name=trace_name,
    # other params such as trace_name, user_id, version, etc
)

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
@observe
def embed_documents(json_path: str):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """
    langfuse_context.update_current_trace(
        session_id=session_name,
    )
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []

    documents = []
    for entry in data:
        # Build a readable content string from the JSON entry
        content = (
            f"Model: {entry.get('model', '')}\n"
            f"Price: {entry.get('price', '')}\n"
            f"Rating: {entry.get('rating', '')}\n"
            f"SIM: {entry.get('sim', '')}\n"
            f"Processor: {entry.get('processor', '')}\n"
            f"RAM: {entry.get('ram', '')}\n"
            f"Battery: {entry.get('battery', '')}\n"
            f"Display: {entry.get('display', '')}\n"
            f"Camera: {entry.get('camera', '')}\n"
            f"Card: {entry.get('card', '')}\n"
            f"OS: {entry.get('os', '')}\n"
            f"In Stock: {entry.get('in_stock', '')}"
        )
        documents.append(Document(page_content=content))

    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient("http://localhost:6333")

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            qdrant_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings_model
            )

            qdrant_store.add_documents(documents=documents)

            return qdrant_store

        # no need to create a vector store every time
        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: The smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        print(f"Error during smartphone information retrieval for model {model}: {e}")
        return f"Error during smartphone information retrieval: {e}"

@tool("EndSession")
def end_session_tool(session_status: str):
    """
    Ends the current session and outputs a goodbye message when the user
    expresses gratitude, or it is clear they would like to end the
    current conversation

    :param
        session_status (str): Status message indicating the end of session
        Should always be "exit"

    :returns
        Exits the system after printing the goodbye message.
    """

    prompt = (
        "You are an AI assistant specialized in answering questions about smartphone features and comparisons. "
        "Provide a polite goodbye message and also thank the user for their feedback."
    )

    try:
        # update the trace
        langfuse_context.update_current_trace(
            session_id=session_name
        )
        # get the current Langfuse handler
        from_context_langfuse_handler = langfuse_context.get_current_langchain_handler()
        config = {
            "callbacks": [from_context_langfuse_handler],
            "run_name": "goodbye",
        }
        goodbye_message = llm.invoke(
            prompt,
            config=config,
        ).content

        # collect feedback
        # retrieve all traces with name `main`
        # associate the score with the latest trace (`traces[0]`) and push score to Langfuse
        traces = langfuse_client.fetch_traces(name="main").data
        trace = langfuse_client.fetch_trace(id=traces[0].id)
        print(trace)
        # [TraceWithDetails(id='7d7f5199-ed3d-439d-9618-cdf21223cc5f', ... createdAt='2025-03-26T08:28:47.870Z')]
        feedback = input("Rate the model's responses from 1 to 5 (e.g., 3 for three stars):")
        comment = input("Please give us a reason for your answer. This will help us improve:")
        langfuse_client.score(
            trace_id=trace.data.id,
            name="usefulness",
            value= float(feedback),
            comment=comment
        )

        print(goodbye_message)

    except Exception as e:
        print("Thank you for visiting. Goodbye!")
    sys.exit(0)


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
def generate_context(llm_tools):
    """
    Process tool calls from the language model and collect their responses.

    :param
        llm_with_tools: The language model instance with bound tools.

    :returns
        list: A list of responses generated by tool calls.
    """
    generated_context = []

    # Process each tool call based on its name
    for tool_call in llm_tools.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_response = smartphone_info_tool.invoke(tool_call)
            generated_context.append(tool_response)
        elif tool_call["name"] == "EndSession":
            tool_response = end_session_tool.invoke(tool_call).content
            generated_context.append(tool_response)
        else:
            generated_context.append("No tool found for this query.")
            sys.exit(0)
    return generated_context

# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe
def main():
    # update the trace
    langfuse_context.update_current_trace(
        session_id=session_name
    )
    # get the current Langfuse handler
    from_context_langfuse_handler = langfuse_context.get_current_langchain_handler()
    config = {
        "callbacks": [from_context_langfuse_handler],
    }
    # List of available tools
    tools = [smartphone_info_tool, end_session_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(
        "You are an expert AI assistant dedicated to helping customers choose the best smartphone from our product catalog. \n"
        "Your sole focus is to provide detailed information about smartphone features and perform comparisons. \n"
        "DO NOT assist with ordering, returns, or general customer support. \n"
        "If a query does not pertain to smartphone features or comparisons, respond that you CANNOT help with that request. \n"
        "When chatting, engage with the user but ensure you only use the smartphone info tool to retrieve specifications from our catalog. \n"
        "NEVER guess or assume a smartphone model based on internal knowledge; always clarify which model the user is referring to. \n\n"
        "Your analysis should always be simple and never exceed 100 words. "
        "When recommending a smartphone, the most important features are: \n"
        " - performance \n -display quality \n -battery life \n -camera capabilities\n"
        " - any special functionalities (e.g., 5G support, fast charging, expandable storage). "
        "Explain how these features translate into real-life benefits for the user, rather than simply listing technical specifications. \n"
        "Clearly state why this phone is a good option, considering these features, but always clarify with the user on what they are looking for. \n"
        "Remember you can check if a product is in stock using context but you can NEVER help with queries related to ordering, support, or others. "
        "You can only assist with smartphone recommendations and comparisons ONLY!"
    )

    conversation = [{"role": "system", "content": system_prompt.content}]  # Initialize conversation with system prompt

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()

            # Format conversation history into a single string for prompt injection
            chat_history = ""
            for message in conversation:
                chat_history += f"{message['role']}: {message['content']}\n"

            prompt = PromptTemplate.from_template(
                f"{chat_history}\n\n"
                f"You have been asked: {user_input}"
            )

            # Append the user's query to the chat history
            conversation.append({"role": "user", "content": HumanMessage(user_input).content})

            chain = prompt | llm_with_tools | generate_context

            config["run_name"] = "context"
            context = chain.invoke(
                {"chat_history": chat_history, "user_input": user_input},
                config=config,
            )

            final_template = PromptTemplate.from_template(
                f"Conversation History:\n{chat_history}\n\n"
                f"User query: {user_input}\n\n"
                f"Context: {context}\n\n"
                "Your analysis should always be simple and never exceed 100 words. "
            )

            config["run_name"] = "final_response"
            response = llm.invoke(
                final_template.invoke({"chat_history": chat_history, "user_input": user_input, "context": context}),
                config=config,
            )
            conversation.append({"role": "assistant", "content": response.content})
            print(f"System: {response.content}")

    except KeyboardInterrupt:
        end_session_tool.invoke({"session_status": "exit"})
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Build the product database vector store
    product_db = embed_documents("smartphones.json")

    main()
