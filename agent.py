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
load_dotenv()

# For simple security in example:
REQUIRED_API_KEY = os.getenv("MY_API_KEY", "my_secret_key")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize CDP AgentKit wrapper
cdp = CdpAgentkitWrapper()
cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(cdp)
tools = cdp_toolkit.get_tools()

# Create the agent
agent_executor = create_react_agent(
    llm,
    tools=tools,
    state_modifier="You are a helpful agent that can interact with the Base blockchain using CDP AgentKit. "
                   "You can create wallets, deploy tokens, and perform transactions."
)

# Initialize Flask
app = Flask(__name__)



##################################### HELPER FUNCTIONS #####################################
# Function to ask the agent a question or perform on-chain actions.
def ask_agent(question: str, session_id: str):
    """
    Generic function to stream the LLM's response based on a prompt (question).
    """
    output_chunks = []
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": session_id}}
    ):
        if "agent" in chunk:
            output_chunks.append(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            output_chunks.append(chunk["tools"]["messages"][0].content)
    return "\n".join(output_chunks)


# Function to generate or update lore based on recent events.
def generate_lore(recent_event: str = "", lore: str = ""):
    """
    Calls the AI with a specific prompt to create or update lore.
    'recent_event' is any prior event text, 'lore' is the last known lore JSON/string.
    """
    lore_prompt = f"""
    You are a game lore generator.
    The current lore is: {lore}
    The most recent event is: {recent_event}
    Please craft or update the lore of a 2D RPG-like grassland town with 2 characters.
    Provide a simple text string of only the updated lore.
    """

    response = ask_agent(lore_prompt, session_id="lore_generation")
    return response



##################################### ROUTES #####################################
# Endpoint to ask the agent a question or perform on-chain actions.
@app.route("/ask", methods=["POST"])
def ask_endpoint():
    """
    General endpoint if you want to do arbitrary questions.
    """
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


# Endpoint to generate or update lore based on recent events.
@app.route("/generate_lore", methods=["POST"])
def generate_lore_endpoint():
    """
    Generates or updates lore. Expects JSON with optional keys:
    - 'recent_event': string
    - 'lore': string (the last known lore)
    Returns as simple text string of only lore.
    """
    api_key = request.headers.get("X-Api-Key")
    if api_key != REQUIRED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json() or {}
    recent_event = data.get("recent_event", '')
    lore = data.get("lore", '')

    lore_text = generate_lore(recent_event, lore)
    return jsonify({"lore": lore_text})


# Endpoint to generate a random event or dialogue based on the current lore.
@app.route("/generate_event", methods=["POST"])
def generate_event_endpoint():
    api_key = request.headers.get("X-Api-Key")
    if api_key != REQUIRED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json() or {}
    current_lore = data.get("lore", "{}")
    game_data = data.get("gameData", "{}")

    event_prompt = f"""You are an event generator for a dynamic RPG game.
    Current Lore: {current_lore}

    Available Game Data:
    {game_data}

    Based on the current lore, choose the most likely event to happen out of the possibilities included in the Game Data, try not to reselect events that have already happened recently, output a valid JSON event using this structure:
    {{
        "location": "<one of the available location IDs>",
        "character": "<an available protagonist character ID or false>",
        "eventType": "<one of the allowed events>",
        "eventExplanation": "<brief explanation of the event>"
    }}
    Only output JSON and nothing else.
    """
    event_text = ask_agent(event_prompt, session_id="lore_events")
    return jsonify({"event": event_text})


# Endpoint to generate random dialogue based on the current lore.
@app.route("/generate_dialogue", methods=["POST"])
def generate_dialogue_endpoint():
    api_key = request.headers.get("X-Api-Key")
    if api_key != REQUIRED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json() or {}
    current_lore = data.get("lore", "{}")

    dialogue_prompt = f"""You are a dialogue generator for this game. 
    Current Lore: {current_lore}
    Create new dialogue. Output JSON:
    {{
    "character": "Alice",
    "dialogue": "Alice whispers: 'I must hide...'"
    }}
    Allowed characters: ["Alice", "Bob", "Charlie", "Daisy", "Eve", "Felix", "Grace", "Hank", "Ivy", "Jack"]
    """
    dialogue_text = ask_agent(dialogue_prompt, session_id="lore_dialogue")
    return jsonify({"dialogue": dialogue_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)