# Web Chatbot — the Flask layer
#
# This module owns everything web: routes, request validation, session cookies,
# JSON responses and HTTP status codes. It contains no model code at all. All
# the chatbot behaviour lives in chatbot_engine.py, which this file imports.
#
# The one job this layer does that the engine cannot is decide *which*
# conversation a request belongs to. It does that with a signed session cookie.
#
# SETUP
# -----
#   pip install flask transformers torch
#
# Run the server:
#   python app.py
#
# Then open http://127.0.0.1:5000 in a browser.

import os                                                              # Read environment variables (the secret key, the port)
import uuid                                                            # Generates a unique ID for each browser session
from flask import Flask, jsonify, render_template, request, session    # The Flask pieces this app needs
from chatbot_engine import ChatbotEngine                               # The web-agnostic chatbot from the sibling module

MAX_MESSAGE_CHARS = 1000  # Reject anything longer — a browser can POST megabytes, and the model would only truncate it anyway

app = Flask(__name__)  # Create the application object; __name__ tells Flask where to look for the templates/ folder
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key")  # Signs the session cookie; a fixed dev default keeps sessions alive across restarts

engine = ChatbotEngine()  # Constructed at import time (instant), but not loaded — load() is called from __main__


# ---------------------------------------------------------------------------
# Session handling — mapping a browser to a conversation
# ---------------------------------------------------------------------------

def get_conversation_id():
    """Return this browser's conversation ID, creating and storing one on first visit."""
    if "sid" not in session:              # No session cookie yet — this is a brand new visitor
        session["sid"] = uuid.uuid4().hex  # Generate a random 32-character ID and store it in the signed cookie
    return session["sid"]                  # The engine uses this string to keep conversations apart


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """Serve the chat page."""
    return render_template("index.html")  # Flask looks for templates/index.html next to this file


@app.get("/health")
def health():
    """Report whether the server is up and the model has finished loading."""
    return jsonify(                                  # Turn the dict into a JSON response with the right Content-Type header
        status="ok",                                 # If this route answers at all, the web server itself is running
        model_loaded=engine.is_ready,                # Distinguishes "server started" from "model ready" while debugging
        active_sessions=engine.active_conversations,  # Handy for confirming that sessions are being created and reset
    )


@app.post("/chat")
def chat():
    """Accept {"message": "..."} as JSON and return {"reply": "..."} as JSON."""
    data = request.get_json(silent=True) or {}  # silent=True returns None instead of raising on malformed JSON, so we can send a clean 400
    message = (data.get("message") or "").strip()  # Missing key and null both become "", which the next check rejects

    if not message:                                                   # Empty or whitespace-only message
        return jsonify(error="Message must not be empty."), 400       # 400 Bad Request — the client sent something invalid
    if len(message) > MAX_MESSAGE_CHARS:                              # Oversized message
        return jsonify(error=f"Message must be at most {MAX_MESSAGE_CHARS} characters."), 400

    if not engine.is_ready:                                                    # A request arrived before load() finished
        return jsonify(error="Model is still loading, try again shortly."), 503  # 503 Service Unavailable — temporary, retry later

    try:
        reply, history_tokens = engine.reply(get_conversation_id(), message)  # Hand the whole turn to the engine
    except Exception as exc:                                       # Any failure inside generation — bad shapes, out of memory, etc.
        app.logger.exception("Generation failed")                  # Log the full traceback to the terminal for debugging
        return jsonify(error=f"Generation failed: {exc}"), 500     # 500 Internal Server Error — the client did nothing wrong

    return jsonify(                        # Send the reply back as JSON for the frontend to render
        reply=reply,
        history_tokens=history_tokens,     # Exposing the token count makes the 512-token trimming visible in the UI
    )


@app.post("/reset")
def reset():
    """Forget this browser's conversation history and start fresh."""
    engine.reset(get_conversation_id())  # Drops the stored history; the browser keeps its session ID
    return jsonify(status="reset")       # Confirm to the frontend that it can clear the transcript


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run when this file is executed directly (not imported by a production server)
    print(f"=== Web Chatbot Tutorial — Flask + {engine.model_name} ===\n")  # Derived from the engine so it never goes stale on a model swap

    engine.load()  # Load the model before accepting any requests, so the first visitor does not wait
    print("Open http://127.0.0.1:5000 in your browser.\n")  # Point the user at the UI

    app.run(
        host="127.0.0.1",   # Listen on localhost only — never expose this dev server to the network
        port=int(os.environ.get("PORT", 5000)),  # Override with PORT=5001 if something else already owns port 5000
        debug=True,         # Show interactive tracebacks in the browser and reload templates/index.html on every request
        use_reloader=False, # Auto-restart would reload the 863 MB model on every save — restart manually instead (see the workshop)
    )
