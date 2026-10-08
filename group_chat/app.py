# Community Chat — the Flask layer
#
# This module owns everything web: routes, request validation, session cookies,
# JSON responses and HTTP status codes. It contains no model code and no
# turn-taking rules. The chat lives in chat_room.py, the model in bot_engine.py.
#
# The one job this layer does that the room cannot is decide *who* a request is
# from. It does that with a signed session cookie holding a display name.
#
# SETUP
# -----
#   pip install flask transformers torch
#
# Run the server:
#   python app.py
#
# Then open http://127.0.0.1:5000 in a browser — and in a second browser window,
# so you can watch two members talk to the same room.

import os                                                              # Read environment variables (the secret key, the port)
from flask import Flask, jsonify, render_template, request, session    # The Flask pieces this app needs
from bot_engine import BotEngine                                       # The model layer
from chat_room import ChatRoom                                         # The room layer, which owns the transcript

MAX_MESSAGE_CHARS = 500  # Group messages are short; a browser can POST megabytes and every extra character costs every bot context

app = Flask(__name__)  # Create the application object; __name__ tells Flask where to look for the templates/ folder
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key")  # Signs the session cookie; a fixed dev default keeps sessions alive across restarts

engine = BotEngine()      # Constructed at import time (instant), but not loaded — load() is called from __main__
room = ChatRoom(engine)   # One room, one process. Every visitor to this server lands in it


# ---------------------------------------------------------------------------
# Identity — mapping a browser to a member of the room
# ---------------------------------------------------------------------------

def current_member():
    """This browser's display name, or None if it has not joined yet."""
    return session.get("name")  # Set by /join; the signed cookie is what stops a visitor claiming to be someone else


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """Serve the chat page."""
    return render_template(               # Flask looks for templates/index.html next to this file
        "index.html",
        model_name=engine.model_name,     # Passed into the template so the page never hardcodes a model name that can go stale
        member=current_member(),          # None means the page opens on the join screen instead of the chat
    )


@app.get("/health")
def health():
    """Report whether the server is up and the model has finished loading."""
    return jsonify(
        status="ok",                        # If this route answers at all, the web server itself is running
        model_loaded=engine.is_ready,       # Distinguishes "server started" from "model ready" while debugging
        members=[m["name"] for m in room.members()],  # Confirms joins are landing where you think they are
    )


@app.post("/join")
def join():
    """Claim a display name and store it in the session cookie."""
    data = request.get_json(silent=True) or {}   # silent=True returns None instead of raising on malformed JSON, so we can send a clean 400
    name = (data.get("name") or "").strip()      # Missing key and null both become "", which the room rejects

    try:
        name = room.join(name)                        # The room owns the rules: length, characters, and collisions with a bot or another member
    except ValueError as exc:                         # Raised for every invalid or taken name
        return jsonify(error=str(exc)), 400           # 400 Bad Request — the client sent something invalid

    session["name"] = name       # From here on this browser is that member
    return jsonify(name=name)


@app.post("/leave")
def leave():
    """Give up this browser's name. The transcript is untouched — it belongs to the room."""
    session.pop("name", None)    # Drop the cookie value; the page falls back to the join screen
    return jsonify(status="left")


@app.get("/messages")
def messages():
    """
    Return everything that happened after ?since=<id>.

    This is the route that makes a *group* chat work. In a one-to-one chatbot
    the reply comes back in the response to the message you sent, so no such
    route is needed. Here most messages are written by someone else — another
    member, or a bot on the worker thread — and polling is how a browser finds
    out about them.
    """
    try:
        since = int(request.args.get("since", 0))     # The highest message id this client has already seen
    except ValueError:                                # ?since=abc
        return jsonify(error="'since' must be an integer."), 400

    return jsonify(room.state_since(since))           # New messages, who is typing, the member list and the epoch


@app.post("/messages")
def send():
    """Accept {"text": "..."} as JSON and add it to the room."""
    member = current_member()
    if not member:                                              # Cookie missing, cleared, or the server restarted with a new secret key
        return jsonify(error="Join the room first."), 401       # 401 Unauthorized — the frontend reopens the join screen on this

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()                     # Missing key, null and whitespace-only all collapse to ""

    if not text:
        return jsonify(error="Message must not be empty."), 400
    if len(text) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"Message must be at most {MAX_MESSAGE_CHARS} characters."), 400

    if not engine.is_ready:                                                       # A request arrived before load() finished
        return jsonify(error="Model is still loading, try again shortly."), 503   # 503 Service Unavailable — temporary, retry later

    message = room.post(member, text)   # Returns as soon as the message is in the log — bot replies are generated on the worker thread
    return jsonify(message.as_dict()), 201  # 201 Created, because this request added a resource to the transcript


@app.post("/reset")
def reset():
    """Clear the transcript — for everybody, not just the caller."""
    if not current_member():                                # Only members may wipe the room
        return jsonify(error="Join the room first."), 401
    room.clear()                                            # Every other browser notices via the epoch in /messages and redraws
    return jsonify(status="cleared")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only run when this file is executed directly (not imported by a production server)
    print(f"=== Community Chat Tutorial — Flask + {engine.model_name} ===\n")  # Derived from the engine so it never goes stale on a model swap

    engine.load()   # Load the model before accepting any requests, so the first visitor does not wait
    room.start()    # Start the worker thread that generates bot replies — without this the bots stay silent forever

    print("Open http://127.0.0.1:5000 in your browser.")
    print("Open it a second time in a private window to join as another member.\n")

    app.run(
        host="127.0.0.1",   # Listen on localhost only — never expose this dev server to the network
        port=int(os.environ.get("PORT", 5000)),  # Override with PORT=5001 if something else already owns port 5000
        debug=True,         # Show interactive tracebacks in the browser and reload templates/index.html on every request
        use_reloader=False, # Auto-restart would reload the model and lose the transcript on every save — restart manually instead
        threaded=True,      # The default, and load-bearing here: polling requests must be served while the worker thread generates
    )
