# Chat Room — the shared transcript, the membership list, and the turn-taking rules
#
# This module owns everything that makes a group chat a group chat: one message
# log that every member reads and writes, who is in the room, which bots should
# answer a given message, and a background worker that generates their replies
# without making the sender wait.
#
# It imports bot_engine, and it never imports Flask. The dependency arrow only
# ever points this way:
#
#   app.py  ->  chat_room.py  ->  bot_engine.py
#
# SETUP
# -----
#   pip install transformers torch
#
# Try it standalone, without a web server:
#   python chat_room.py

import queue                                                # A thread-safe FIFO of pending bot turns
import re                                                   # Used to find @mentions and bare names in a message
import threading                                            # The background worker and the lock over the transcript
import time                                                 # Timestamps on messages, and the sleep in the terminal demo
from bot_engine import BotEngine, DEFAULT_PERSONAS          # The model layer — this module's only dependency


class Message:
    """One line of the transcript. Immutable once created."""

    def __init__(self, id, speaker, text, is_bot):
        self.id = id            # Monotonically increasing integer — the whole polling protocol is built on it
        self.speaker = speaker  # Display name, human or bot; the transcript does not care which
        self.text = text        # What was said
        self.is_bot = is_bot    # Drives the "BOT" tag in the UI and the loop-prevention rules below
        self.ts = time.time()   # Unix timestamp, formatted by the frontend

    def as_dict(self):
        """The wire format. Defined here so the room, not app.py, decides what a message looks like."""
        return {"id": self.id, "speaker": self.speaker, "text": self.text, "is_bot": self.is_bot, "ts": self.ts}


class ChatRoom:
    """
    A single shared chat room with human and bot members.

    Every public method is safe to call from several threads at once, which is
    not optional here: Flask handles requests in parallel and the room runs a
    worker thread of its own.

    Typical use:
        room = ChatRoom(engine)
        room.start()
        room.join("Sam")
        room.post("Sam", "hello @Ada")   # returns immediately; Ada answers later
    """

    def __init__(
        self,
        engine,
        personas=DEFAULT_PERSONAS,
        context_messages=20,  # How much of the transcript a bot is shown — the recent past, not the whole history
        max_bot_chain=2,      # How many bot messages may follow one another before a human has to speak again
    ):
        self._engine = engine                                    # The model layer; the room never touches torch itself
        self._bots = {p.name.lower(): p for p in personas}       # Lower-cased name -> Persona, so lookups are case-insensitive
        self.context_messages = context_messages                 # Passed to the engine as a slice of the log
        self.max_bot_chain = max_bot_chain                       # The loop brake — see _wake()

        self._messages = []      # The shared transcript, oldest first
        self._next_id = 1        # Never reset, not even by clear() — see the note there
        self._humans = {}        # Lower-cased name -> display name of every human who has joined
        self._pending = []       # Bots queued or currently generating, in order; shown as "typing…"
        self._bot_chain = 0      # Consecutive bot messages since the last human one
        self._epoch = 1          # Bumped by clear(); lets clients notice someone wiped the room

        self._lock = threading.RLock()  # Guards every attribute above. Re-entrant because _wake() is called with it already held
        self._queue = queue.Queue()     # Bot turns waiting to be generated, as (persona_name, epoch) pairs
        self._worker = None             # The daemon thread created by start()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        """Start the background worker. Call once, after the engine has loaded."""
        if self._worker is not None:                    # Guard against a double start, e.g. from a reloader
            return
        self._worker = threading.Thread(
            target=self._work,
            name="bot-worker",
            daemon=True,  # Daemon threads do not keep the process alive, so Ctrl+C still exits immediately
        )
        self._worker.start()

    # -----------------------------------------------------------------------
    # Membership
    # -----------------------------------------------------------------------

    def join(self, name):
        """
        Register a human display name and return it.

        Raises ValueError if the name is unusable or already taken — the caller
        turns that into an HTTP error.
        """
        name = " ".join(name.split())                       # Collapse runs of whitespace; "  Sam  Lee " -> "Sam Lee"
        if not 1 <= len(name) <= 24:
            raise ValueError("Name must be between 1 and 24 characters.")
        if not re.fullmatch(r"[\w .'-]+", name):            # Letters, digits, underscore plus a few name characters
            raise ValueError("Name may only contain letters, numbers, spaces and . ' -")

        with self._lock:
            key = name.lower()                              # Names are compared case-insensitively so "ada" cannot impersonate "Ada"
            if key in self._bots:
                raise ValueError(f"{name} is one of the bots — pick another name.")
            if key in self._humans and self._humans[key] != name:
                raise ValueError(f"{name} is already taken.")
            self._humans[key] = name
            return name

    def members(self):
        """Everyone in the room, bots first, as plain dicts for the UI."""
        with self._lock:
            bots = [{"name": p.name, "description": p.description, "is_bot": True} for p in self._bots.values()]
            humans = [{"name": n, "description": "", "is_bot": False} for n in self._humans.values()]
            return bots + humans

    def _member_names(self):
        """Every name in the room. The caller must hold self._lock."""
        return [p.name for p in self._bots.values()] + list(self._humans.values())

    # -----------------------------------------------------------------------
    # Reading and writing the transcript
    # -----------------------------------------------------------------------

    def post(self, speaker, text):
        """
        Add a human message and wake whichever bots should answer it.

        Returns the stored Message immediately. The bots' replies are generated
        on the worker thread and appear in the log seconds later.
        """
        with self._lock:
            message = self._append(speaker, text, is_bot=False)  # Append first, so the sender's own message is never delayed by policy
            self._wake(message)                                  # Then decide who answers; still under the lock, so the log cannot shift underneath
            return message

    def messages_since(self, last_id):
        """Every message with an id greater than last_id, oldest first."""
        with self._lock:
            return [m for m in self._messages if m.id > last_id]  # A linear scan is fine at this size; an index would be premature

    def state_since(self, last_id):
        """One snapshot for the polling endpoint: new messages plus everything the UI redraws."""
        with self._lock:
            return {
                "messages": [m.as_dict() for m in self._messages if m.id > last_id],
                "typing": list(self._pending),   # Bots the UI should show as writing
                "members": self.members(),       # Cheap enough to resend, and it keeps the member list live
                "epoch": self._epoch,            # If this changed, someone cleared the room and the client must drop its transcript
            }

    def clear(self):
        """Wipe the transcript for everyone. Shared state means this is not a private action."""
        with self._lock:
            self._messages = []
            self._bot_chain = 0
            self._epoch += 1  # Invalidates work already queued (the worker checks) and tells clients to redraw from scratch
            # self._next_id deliberately keeps counting. A client that polls with
            # since=42 must never be handed a *new* message numbered 7, or it
            # would decide it had already seen it and drop it.

    def _append(self, speaker, text, is_bot):
        """Create, number and store one message. The caller must hold self._lock."""
        message = Message(self._next_id, speaker, text.strip(), is_bot)
        self._next_id += 1
        self._messages.append(message)
        return message

    # -----------------------------------------------------------------------
    # Turn taking — who speaks next, and when the bots have to stop
    # -----------------------------------------------------------------------

    def _mentioned_bots(self, text):
        """Bots named in a message, either as @Ada or as a bare word. Order follows the member list."""
        words = {w.lower() for w in re.findall(r"@?([\w'-]+)", text)}  # Split into words, dropping any leading @
        return [p for key, p in self._bots.items() if key in words]

    def _least_recently_active(self):
        """The bot that has gone longest without speaking, so an unaddressed message still gets one answer."""
        spoken = []
        for message in reversed(self._messages):                       # Walk backwards: most recent speaker first
            key = message.speaker.lower()
            if message.is_bot and key not in spoken:
                spoken.append(key)
        for key in self._bots:                                          # Any bot that has never spoken wins outright
            if key not in spoken:
                return self._bots[key]
        return self._bots[spoken[-1]]                                   # Otherwise the one furthest back in the log

    def _wake(self, message):
        """
        Queue the bots that should respond to `message`.

        This is the whole social policy of the room, and the reason it exists is
        that bots answering bots is a loop with no natural end: two chatty
        personas will happily fill the log until the process is killed. Three
        rules stop it, and all three matter:

          1. A bot never answers itself.
          2. A bot answers *another bot* only when that bot named it.
          3. At most max_bot_chain bot messages may follow one human message.

        The caller must hold self._lock.
        """
        if message.is_bot:
            self._bot_chain += 1                     # One more bot message since a human last spoke
            if self._bot_chain >= self.max_bot_chain:  # The brake: the chain has run long enough
                return                                 # A human has to say something before the bots continue
            targets = self._mentioned_bots(message.text)  # Bot to bot requires an explicit mention
        else:
            self._bot_chain = 0                                            # A human speaking always resets the brake
            targets = self._mentioned_bots(message.text)                   # Addressed bots answer...
            targets = targets or [self._least_recently_active()]           # ...and if nobody was addressed, exactly one bot picks it up

        for persona in targets:
            if persona.name == message.speaker:      # Rule 1 — never reply to yourself
                continue
            if persona.name in self._pending:        # Already queued or generating; queuing twice would double-post
                continue
            self._pending.append(persona.name)       # The UI reads this as "Ada is typing…"
            self._queue.put((persona.name, self._epoch))  # Hand the turn to the worker thread, tagged with the epoch it was queued in

    # -----------------------------------------------------------------------
    # The background worker
    # -----------------------------------------------------------------------

    def _work(self):
        """
        Generate queued bot messages, one at a time, forever.

        This runs on its own thread for one reason: a reply takes 10-15 seconds
        on a CPU, and the member who typed must not sit through it. Their POST
        returns as soon as their own message is in the log; the bot's answer
        arrives in a later poll.

        One worker, not several. The model is a single shared resource that the
        engine serialises anyway, so extra workers would queue on the same lock
        while making the transcript order much harder to reason about.
        """
        while True:
            persona_name, epoch = self._queue.get()   # Blocks until there is a turn to take
            try:
                self._speak(persona_name, epoch)
            except Exception as exc:                                        # A crash here would silently kill the worker and every future reply
                print(f"[bot-worker] {persona_name} failed: {exc!r}")       # Log it and keep the thread alive
            finally:
                with self._lock:
                    if persona_name in self._pending:
                        self._pending.remove(persona_name)  # Clear the "typing" flag whatever happened, or it sticks forever

    def _speak(self, persona_name, epoch):
        """Generate one persona's message and append it. Runs on the worker thread."""
        with self._lock:                                              # Take a consistent snapshot...
            if epoch != self._epoch:                                  # ...unless the room was cleared while this turn waited in the queue
                return                                                # Drop it: the conversation it was answering no longer exists
            persona = self._bots[persona_name.lower()]
            recent = self._messages[-self.context_messages:]          # Only the recent past goes to the model
            transcript = [{"speaker": m.speaker, "text": m.text} for m in recent]  # Plain dicts — the engine must not see Message objects
            names = self._member_names()

        if not transcript:                                            # Nothing to answer, e.g. cleared between the check and here
            return

        text, _tokens = self._engine.reply(persona, transcript, names)  # The slow part — 10-15s — runs with the lock released

        with self._lock:
            if epoch != self._epoch:            # Check again: the room may have been cleared while the model was generating
                return
            message = self._append(persona.name, text, is_bot=True)
            self._wake(message)                 # A bot message can wake another bot, subject to the rules in _wake()


# ---------------------------------------------------------------------------
# Terminal demo — proof that the room does not need a web server
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Only runs when this file is executed directly, not when app.py imports it
    print("=== Chat Room — terminal demo ===\n")

    engine = BotEngine()
    engine.load()

    room = ChatRoom(engine)
    room.start()            # Without this, messages are queued and nothing ever answers
    me = room.join("Sam")

    print("\nYou are Sam. Mention a bot by name to address it (@Ada, @Newton, @Pixel).")
    print("Bot replies arrive a few seconds after your message. Type 'quit' to exit.\n")
    print("In the room:", ", ".join(m["name"] for m in room.members()), "\n")

    seen = 0  # The id of the last message printed — the same protocol the browser uses
    while True:
        text = input(f"{me}: ").strip()
        if not text:
            continue
        if text.lower() in ("quit", "stop", "exit"):
            print("\nLeaving the room.\n")
            break

        room.post(me, text)  # Returns immediately, before any bot has answered

        # Poll until the room goes quiet, the way the browser frontend does.
        # A blocking input() would hide bot messages until the next keystroke.
        while True:
            for message in room.messages_since(seen):
                seen = message.id
                if message.speaker != me:                        # Skip the echo of what was just typed
                    print(f"  {message.speaker}: {message.text}")
            with room._lock:                                      # Reaching into a private attribute is acceptable in a demo, not in app.py
                busy = bool(room._pending)
            if not busy:
                break
            time.sleep(0.5)
        print()
