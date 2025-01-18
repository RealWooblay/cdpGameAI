##################################### IMPORTS #####################################
# CDP and OpenAI Imports
from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Server Imports
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv


##################################### INITIALIZATION #####################################
# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize CDP AgentKit wrapper
cdp = CdpAgentkitWrapper()

# Create toolkit from wrapper
cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(cdp)

# Get all available tools
tools = cdp_toolkit.get_tools()


# Create the agent
agent_executor = create_react_agent(
    llm,
    tools=tools,
    state_modifier="You are a helpful agent that can interact with the Base blockchain using CDP AgentKit. You can create wallets, deploy tokens, and perform transactions."
)

# Initialize the Flask app
app = Flask(__name__)

##################################### MAIN #####################################

# Function to interact with the agent with multiple sessions on Flask server
def ask_agent(question: str, session_id: str):
    output_chunks = []

    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": session_id}}
    ):
        if "agent" in chunk:
            output_chunks.append(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            output_chunks.append(chunk["tools"]["messages"][0].content)

    # Return the combined output as a single string (or structured how you want)
    return "\n".join(output_chunks)

# Endpoint to interact with the agent
@app.route("/ask", methods=["POST"])
def ask_endpoint():
    """
    Example endpoint:
      - Expects JSON with { "session_id": "<some_id>", "question": "..." }
      - Returns the agent's response
    """
    # Security check: require X-Api-Key or similar
    api_key = request.headers.get("X-Api-Key")
    if api_key != REQUIRED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.get_json() or {}
    question = data.get("question", "")
    session_id = data.get("session_id", "default_session")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    agent_response = ask_agent(question, session_id)
    return jsonify({"response": agent_response})

# Run the AI agent server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)    # TODO: For production, consider using gunicorn/uwsgi. 