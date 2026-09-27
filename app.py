import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

LEON_SYSTEM = """
You are Leon, a personal AI assistant.

Personality:
- Friendly, calm and natural.
- Speak like a helpful personal assistant, not like a robot.
- Understand normal conversational language.
- Be honest when you do not know something.
- Never invent facts just to sound confident.
- Explain difficult things simply.
- Keep answers appropriate and useful for the user.
- Remember the conversation history provided to you.
- Do not claim that you performed an action unless the system actually performed it.

Your role:
- Help with study, coding, general knowledge, planning and everyday questions.
- Give clear answers first, then explanation when useful.
- If information is uncertain or may have changed, clearly say so.
"""

@app.get("/")
def home():
    return jsonify({
        "name": "Leon API",
        "status": "online",
        "version": "1.0"
    })


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({
            "error": "Message is required"
        }), 400

    messages = []

    for item in history[-20:]:
        role = item.get("role")
        content = item.get("content")

        if role in ["user", "assistant"] and content:
            messages.append({
                "role": role,
                "content": content
            })

    messages.append({
        "role": "user",
        "content": message
    })

    try:
        response = client.responses.create(
            model=os.getenv("LEON_MODEL", "gpt-5.6-luna"),
            instructions=LEON_SYSTEM,
            input=messages
        )

        reply = response.output_text

        return jsonify({
            "reply": reply
        })

    except Exception as e:
        return jsonify({
            "error": "Leon could not process the request."
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port
    )
