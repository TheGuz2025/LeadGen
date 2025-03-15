import openai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_openai():
    prompt = "What is the capital of France?"
    
    try:
        response = openai.chat.completions.create(  # ✅ Updated OpenAI function
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        # ✅ Extract AI response correctly
        ai_response = response.choices[0].message.content
        print("✅ OpenAI API Response:", ai_response)

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")

# Run the test
test_openai()
