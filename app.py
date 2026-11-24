import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Temperature mapping based on Genre (0.0 to 1.0)
GENRE_TEMPERATURE = {
    "Comedy": 0.9,
    "Fantasy": 0.85,
    "Sci-Fi": 0.8,
    "Adventure": 0.75,
    "Romance": 0.7,
    "Horror": 0.7,
    "Mystery": 0.5,
    "Historical": 0.3,
}

# Max Tokens mapping based on Story Length
LENGTH_SETTINGS = {
    "Short": {"max_tokens": 300, "desc": "short (around 150 words)"},
    "Medium": {"max_tokens": 700, "desc": "medium-length (around 400 words)"},
    "Long": {"max_tokens": 1400, "desc": "long, detailed (around 800 words)"},
}

class StoryRequest(BaseModel):
    topic: str
    genre: str
    character_name: str = ""
    length: str

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/generate-story")
async def generate_story(req: StoryRequest):
    temperature = GENRE_TEMPERATURE.get(req.genre, 0.7)
    length_cfg = LENGTH_SETTINGS.get(req.length, LENGTH_SETTINGS["Medium"])
    max_tokens = length_cfg["max_tokens"]

    char_name = req.character_name.strip()
    if char_name:
        character_instruction = f"The main character's name is {char_name}."
    else:
        character_instruction = f"Invent an engaging, fitting name for the main character that suits a {req.genre} story."

    prompt = (
        f"Write a {length_cfg['desc']} {req.genre} story about '{req.topic}'. "
        f"{character_instruction} "
        f"Give the story a catchy title at the top and make sure it has a complete ending."
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a master storyteller. Write engaging, well-paced stories matching the requested genre and length."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="openai/gpt-oss-20b",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    story = chat_completion.choices[0].message.content

    return JSONResponse({
        "story": story,
        "temperature_used": temperature,
        "max_tokens_used": max_tokens
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)