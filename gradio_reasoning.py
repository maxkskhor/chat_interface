import os
import time

import gradio as gr
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from gradio import ChatInterface


class SideBarChatInterface(ChatInterface):
    """
    Chat Interface with history in the sidebar
    """
    def _render_history_area(self):
        with gr.Sidebar():
            gr.Markdown('# Chat AI')
            self.new_chat_button = gr.Button(
                "New chat",
                variant="primary",
                size="md",
                icon=gr.utils.get_icon_path("plus.svg"),
                # scale=0,
            )
            self.chat_history_dataset = gr.components.Dataset(
                components=[gr.Textbox(visible=False)],
                show_label=False,
                layout="table",
                type="index",
            )

# Load environment variables from .env file
# Make sure you have a .env file with DEEPSEEK_API_KEY=your_key
load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-reasoner"  # Or "deepseek-coder" if you prefer
SYSTEM_PROMPT = "You are a helpful and friendly chatbot."  # Customize your bot's personality

# --- Error Handling ---
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables. "
                     "Please create a .env file with your key.")

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    exit()


# --- Streaming Chat Function ---
def chat_stream(message, history):
    """
    Send a message to DeepSeek API and stream the response.

    Args:
        message (str): Current user message
        history (list): List of [user_message, bot_message] pairs

    Yields:
        str: Streamed fragments of the response
    """
    # Format the conversation history for the API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in history:
        messages.append(i)

    # Add the current message
    messages.append({"role": "user", "content": message})

    try:
        # Call the API with streaming enabled
        # noinspection PyTypeChecker
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,  # Adjust creativity (0.0 to 1.0)
            max_tokens=1000,  # Maximum length of response
            stream=True  # Enable streaming
        )

        # Initialize the response text
        reasoning_content = "[think]"
        content = ""

        for chunk in stream:
            if chunk.choices:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                    yield reasoning_content

                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                    yield reasoning_content + "[/think]" + content

        logger.info(f'reasoning_content: {reasoning_content}')
        logger.info(f'content: {content}')

    except Exception as e_:
        yield f"An error occurred: {str(e_)}"


# --- Streaming Chat Function ---
def chat_stream_with_thinking_box(message, history):
    """
    Send a message to DeepSeek API and stream the response.

    Args:
        message (str): Current user message
        history (list): List of [user_message, bot_message] pairs

    Yields:
        str: Streamed fragments of the response
    """
    # Format the conversation history for the API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in history:
        messages.append(i)

    # Add the current message
    messages.append({"role": "user", "content": message})

    try:
        # Call the API with streaming enabled
        # noinspection PyTypeChecker
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,  # Adjust creativity (0.0 to 1.0)
            max_tokens=1000,  # Maximum length of response
            stream=True  # Enable streaming
        )

        # Initialize the response text
        reasoning_content = ""
        content = ""
        elapsed_thinking = 0

        start_thinking = time.time()
        for chunk in stream:
            if chunk.choices:
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                    elapsed_thinking = time.time() - start_thinking
                    yield [
                        {"role": "assistant",
                         "content": reasoning_content,
                         "metadata": {"title":  "🧠 Thinking", "status": "pending"}}
                    ]
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                    yield [
                        {"role": "assistant",
                         "content": reasoning_content,
                         "metadata": {"title":  "🧠 Thinking", "status": "done", "duration": elapsed_thinking}},
                        {"role": "assistant", "content": content}
                    ]

        logger.info(f'reasoning_content: {reasoning_content}')
        logger.info(f'content: {content}')

    except Exception as e_:
        yield f"An error occurred: {str(e_)}"


# --- Gradio Interface ---
chat_interface = SideBarChatInterface(
    chat_stream_with_thinking_box,
    type='messages',
    title="Thinking Chatbot",
    description="Chat with Reasoning Model.",
    examples=["Hello",
              "Can you write a short poem about technology?",
              "Explain quantum computing in simple terms"],
    run_examples_on_click=False,
    # example_labels=["eg1", "eg2", "eg3"],
    save_history=True,
    flagging_mode='manual',
    flagging_options=('Like', 'Dislike'),
)

# --- Run the app ---
if __name__ == "__main__":
    logger.info("Starting Chat interface with reasoning model...")
    chat_interface.launch()  # Set share=False if you don't want a public URL
