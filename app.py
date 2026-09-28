import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 初始化 Gemini 客户端
# 注意：需在 Render 环境变量中配置 GEMINI_API_KEY
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# 助手设定与系统提示词（设定 Leon 的人设）
LEON_SYSTEM_INSTRUCTION = """
You are Leon, a helpful, polite, intelligent, and natural conversational AI assistant.

Personality & Rules:
- Friendly, empathetic, and clear.
- Provide direct answers first, followed by clear explanations when useful.
- Converse naturally like a supportive companion.
- Maintain context from the ongoing conversation history.
"""

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Leon API (Gemini Powered)",
        "status": "online",
        "version": "1.0"
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # 构建聊天上下文格式
    contents = []

    # 载入历史记录（最多截取前 20 条避免超限）
    for item in history[-20:]:
        role = item.get("role")
        content = item.get("content")
        if role and content:
            # 转换角色格式为 gemini 兼容格式 (user / model)
            mapped_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=mapped_role,
                    parts=[types.Part.from_text(text=content)]
                )
            )

    # 加入当前用户最新消息
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        )
    )

    try:
        # 调用 Gemini API 生成对话回复
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=LEON_SYSTEM_INSTRUCTION,
                temperature=0.7
            )
        )

        reply_text = response.text if response.text else "抱歉，暂时未能生成回复。"

        # 始终返回包含 'reply' 的标准 JSON
        return jsonify({
            "reply": reply_text
        })

    except Exception as e:
        return jsonify({
            "error": "Leon could not process the request.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
