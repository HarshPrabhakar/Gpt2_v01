# D:\Gpt2_v01\app\chat_ui.py

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from app.inference_engine import MyGPTInferenceEngine


# ============================================================
# Load Model Once
# ============================================================

print()
print("=" * 72)
print("MYGPT2 INSTRUCT UI")
print("=" * 72)
print("Loading model...")

engine = MyGPTInferenceEngine()

status = engine.status()

print()
print("Model loaded successfully.")
print(
    f"Checkpoint      : "
    f"{status['checkpoint']}"
)
print(
    f"Checkpoint step : "
    f"{status['step']}"
)
print(
    f"Parameters      : "
    f"{status['parameters']:,}"
)
print(
    f"Context         : "
    f"{status['context_length']}"
)
print(
    f"Vocabulary      : "
    f"{status['vocab_size']:,}"
)
print(
    f"Device          : "
    f"{status['device']}"
)
print(
    f"GPU             : "
    f"{status['gpu']}"
)
print("=" * 72)
print()


# ============================================================
# History Helpers
# ============================================================

def normalize_history(
    history,
):

    if history is None:

        return []

    normalized = []

    for item in history:

        if not isinstance(
            item,
            dict,
        ):

            continue

        role = str(
            item.get(
                "role",
                "",
            )
        ).strip().lower()

        content = item.get(
            "content",
            "",
        )

        if content is None:

            continue

        content = str(
            content
        ).strip()

        if (
            role
            not in {
                "user",
                "assistant",
                "system",
            }
            or
            not content
        ):

            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized


# ============================================================
# Generate Response
# ============================================================

def generate_response(
    message,
    history_state,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    max_new_tokens,
):

    message = (
        ""
        if message is None
        else str(
            message
        ).strip()
    )

    history = normalize_history(
        history_state
    )

    if not message:

        return (
            history,
            history,
            "",
            "Waiting for a prompt...",
        )


    try:

        # ----------------------------------------------------
        # Generate using SFT model and existing conversation
        # ----------------------------------------------------

        result = engine.generate(
            prompt=message,

            history=history,

            temperature=float(
                temperature
            ),

            top_k=int(
                top_k
            ),

            top_p=float(
                top_p
            ),

            repetition_penalty=float(
                repetition_penalty
            ),

            no_repeat_ngram_size=int(
                no_repeat_ngram_size
            ),

            max_new_tokens=int(
                max_new_tokens
            ),
        )


        # ----------------------------------------------------
        # Model response
        # ----------------------------------------------------

        generated_text = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        if not generated_text:

            generated_text = (
                "[Model produced no visible response.]"
            )


        # ----------------------------------------------------
        # Messages-format conversation
        # ----------------------------------------------------

        updated_history = history + [

            {
                "role": "user",
                "content": message,
            },

            {
                "role": "assistant",
                "content": generated_text,
            },

        ]


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        generated_tokens = int(
            result.get(
                "generated_tokens",
                0,
            )
        )

        input_tokens = int(
            result.get(
                "input_tokens",
                0,
            )
        )

        elapsed_seconds = float(
            result.get(
                "elapsed_seconds",
                0.0,
            )
        )

        tokens_per_second = float(
            result.get(
                "tokens_per_second",
                0.0,
            )
        )

        stopped_on_eos = bool(
            result.get(
                "stopped_on_eos",
                False,
            )
        )

        stop_reason = (
            "EOS"
            if result["stopped_on_eos"]
            else "Max tokens"
        )

        metrics = (
            f"Context {result['input_tokens']} tokens"
            f"  •  "
            f"Generated {result['generated_tokens']} tokens"
            f"  •  "
            f"{result['tokens_per_second']:.1f} tok/s"
            f"  •  "
            f"{result['elapsed_seconds']:.2f}s"
            f"  •  "
            f"Stop: {stop_reason}"
            f"  •  "
            f"History: {result['history_turns_used']}/"
            f"{result['history_turns_total']} turns"
            f"  •  "
            f"Dropped: {result['history_turns_dropped']}"
        )


        # ----------------------------------------------------
        # Return
        #
        # 1. Chatbot display
        # 2. Internal state
        # 3. Clear textbox
        # 4. Metrics
        # ----------------------------------------------------

        return (
            updated_history,
            updated_history,
            "",
            metrics,
        )


    except Exception as error:

        error_text = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        print()
        print("=" * 72)
        print("GENERATION ERROR")
        print("=" * 72)
        print(error_text)
        print("=" * 72)
        print()

        return (
            history,
            history,
            message,
            (
                "⚠️ **Generation failed:** "
                f"`{error_text}`"
            ),
        )


# ============================================================
# Clear Conversation
# ============================================================

def clear_chat():

    return (
        [],
        [],
        "",
        "Conversation cleared.",
    )


# ============================================================
# CSS
# ============================================================

CSS = """
:root {
    --background-fill-primary: #080a0f;
    --background-fill-secondary: #0e1118;
    --border-color-primary: #242936;
    --body-text-color: #e7eaf0;
    --body-text-color-subdued: #8f96a6;
}

body {
    background:
        radial-gradient(
            circle at top left,
            rgba(55, 65, 95, 0.16),
            transparent 34%
        ),
        radial-gradient(
            circle at top right,
            rgba(55, 42, 82, 0.12),
            transparent 32%
        ),
        #07090d !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    background: transparent !important;
}

#app-shell {
    border: 1px solid #202532;
    border-radius: 22px;

    overflow: hidden;

    background:
        linear-gradient(
            145deg,
            rgba(18, 21, 29, 0.98),
            rgba(9, 11, 16, 0.98)
        );

    box-shadow:
        0 30px 80px
        rgba(0, 0, 0, 0.45);
}

#header {
    padding:
        10px
        4px
        18px
        4px;
}

#brand-title {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: #f2f4f8;
    margin-bottom: 2px;
}

#brand-subtitle {
    color: #858c9b;
    font-size: 14px;
}

#model-pill {
    border:
        1px solid
        #262c38;

    background:
        #10131a;

    border-radius:
        14px;

    padding:
        12px
        14px;

    color:
        #a9b0bd;

    font-size:
        13px;
}

#sidebar {
    border-right:
        1px solid
        #202532;

    padding-right:
        18px;
}

#settings-title {
    font-weight:
        600;

    margin-bottom:
        4px;
}

#chatbot {
    border:
        1px solid
        #202532 !important;

    border-radius:
        18px !important;

    background:
        #0b0e14 !important;

    min-height:
        610px;
}

#chatbot .message {
    border-radius:
        16px !important;
}

#prompt-box textarea {
    background:
        #0c0f15 !important;

    border:
        1px solid
        #272c38 !important;

    border-radius:
        16px !important;

    color:
        #f0f2f5 !important;

    font-size:
        15px !important;
}

#prompt-box textarea:focus {
    border-color:
        #4a5263 !important;

    box-shadow:
        0 0 0 1px
        #4a5263 !important;
}

#generate-button {
    background:
        linear-gradient(
            135deg,
            #e8ebf0,
            #b9c0cc
        ) !important;

    color:
        #090b10 !important;

    border:
        none !important;

    font-weight:
        700 !important;

    border-radius:
        12px !important;
}

#generate-button:hover {
    filter:
        brightness(
            1.06
        );
}

#clear-button {
    border:
        1px solid
        #292f3b !important;

    background:
        #11151c !important;

    color:
        #aeb4c0 !important;

    border-radius:
        12px !important;
}

#metrics {
    color:
        #7f8796;

    font-size:
        12px;

    margin-top:
        5px;
}

footer {
    display:
        none !important;
}
"""


# ============================================================
# Theme
# ============================================================

theme = gr.themes.Base(

    primary_hue="slate",

    secondary_hue="gray",

    neutral_hue="slate",

).set(

    body_background_fill=(
        "#07090d"
    ),

    block_background_fill=(
        "#0e1118"
    ),

    block_border_color=(
        "#202532"
    ),

    input_background_fill=(
        "#0c0f15"
    ),

    button_primary_background_fill=(
        "#d9dde5"
    ),

    button_primary_text_color=(
        "#0b0d12"
    ),
)


# ============================================================
# User Interface
# ============================================================

with gr.Blocks(
    title="MyGPT2 Instruct",
) as demo:


    # --------------------------------------------------------
    # Internal application state
    #
    # Chatbot = visual component
    # State   = conversation source of truth
    # --------------------------------------------------------

    conversation_state = gr.State(
        []
    )


    with gr.Column(
        elem_id="app-shell",
    ):


        # ====================================================
        # Header
        # ====================================================

        with gr.Row(
            elem_id="header",
        ):


            with gr.Column(
                scale=3,
            ):

                gr.HTML(
                    """
                    <div id="brand-title">
                        MyGPT2 Instruct
                    </div>

                    <div id="brand-subtitle">
                        Local 110M parameter
                        instruction-tuned transformer
                    </div>
                    """
                )


            with gr.Column(
                scale=2,
            ):

                gr.HTML(
                    f"""
                    <div id="model-pill">

                        <b>Checkpoint</b>
                        &nbsp;
                        {status['checkpoint']}

                        &nbsp;&nbsp;·&nbsp;&nbsp;

                        <b>Step</b>
                        &nbsp;
                        {status['step']}

                        &nbsp;&nbsp;·&nbsp;&nbsp;

                        <b>Parameters</b>
                        &nbsp;
                        {status['parameters']/ 1_000_000:.1f}M

                        &nbsp;&nbsp;·&nbsp;&nbsp;

                        <b>GPU</b>
                        &nbsp;
                        {status['gpu']}

                    </div>
                    """
                )


        # ====================================================
        # Main Area
        # ====================================================

        with gr.Row():


            # =================================================
            # Sidebar
            # =================================================

            with gr.Column(
                scale=1,
                elem_id="sidebar",
            ):


                gr.Markdown(
                    "### Generation Settings",
                    elem_id="settings-title",
                )


                temperature = gr.Slider(

                    minimum=0.1,

                    maximum=1.5,

                    value=0.70,

                    step=0.05,

                    label="Temperature",

                    info=(
                        "Lower = more focused, "
                        "higher = more random"
                    ),
                )


                top_k = gr.Slider(

                    minimum=0,

                    maximum=200,

                    value=40,

                    step=1,

                    label="Top-k",
                )


                top_p = gr.Slider(

                    minimum=0.10,

                    maximum=1.0,

                    value=0.90,

                    step=0.01,

                    label="Top-p",
                )


                repetition_penalty = gr.Slider(

                    minimum=1.0,

                    maximum=1.5,

                    value=1.15,

                    step=0.01,

                    label="Repetition penalty",

                    info=(
                        "Discourages repeated tokens"
                    ),
                )


                no_repeat_ngram_size = gr.Slider(

                    minimum=0,

                    maximum=6,

                    value=3,

                    step=1,

                    label="No-repeat n-gram",

                    info=(
                        "3 prevents repeated trigrams; "
                        "0 disables it"
                    ),
                )


                max_new_tokens = gr.Slider(

                    minimum=10,

                    maximum=400,

                    value=120,

                    step=10,

                    label="Max new tokens",
                )


                gr.Markdown(
                    f"""
                    ---

                    ### Model

                    **Architecture**  
                    GPT-2 style decoder

                    **Checkpoint**  
                    `{status['checkpoint']}`

                    **Checkpoint step**  
                    `{status['step']}`

                    **Context**  
                    `{status['context_length']}` tokens

                    **Vocabulary**  
                    `{status['vocab_size']:,}`

                    **Parameters**  
                    `{status['parameters']:,}`

                    **Device**  
                    `{status['device']}`

                    ---

                    ### Recommended baseline

                    Temperature: `0.70`  
                    Top-k: `40`  
                    Top-p: `0.90`  
                    Repetition penalty: `1.15`  
                    No-repeat n-gram: `3`
                    """
                )


            # =================================================
            # Chat
            # =================================================

            with gr.Column(
                scale=4,
            ):


                chatbot = gr.Chatbot(

                    value=[],

                    elem_id="chatbot",

                    height=610,

                    show_label=False,

                    placeholder=(
                        "Start a conversation "
                        "with MyGPT2 Instruct."
                    ),
                )


                message = gr.Textbox(

                    placeholder=(
                        "Message MyGPT2..."
                    ),

                    lines=3,

                    max_lines=8,

                    show_label=False,

                    elem_id="prompt-box",
                )


                with gr.Row():


                    clear_button = gr.Button(

                        "Clear",

                        elem_id="clear-button",
                    )


                    generate_button = gr.Button(

                        "Generate",

                        variant="primary",

                        elem_id="generate-button",
                    )


                metrics = gr.Markdown(

                    "Model ready.",

                    elem_id="metrics",
                )


        # ====================================================
        # Generate Button
        # ====================================================

        generate_button.click(

            fn=generate_response,

            inputs=[

                message,

                conversation_state,

                temperature,

                top_k,

                top_p,

                repetition_penalty,

                no_repeat_ngram_size,

                max_new_tokens,

            ],

            outputs=[

                chatbot,

                conversation_state,

                message,

                metrics,

            ],
        )


        # ====================================================
        # Enter / Submit
        # ====================================================

        message.submit(

            fn=generate_response,

            inputs=[

                message,

                conversation_state,

                temperature,

                top_k,

                top_p,

                repetition_penalty,

                no_repeat_ngram_size,

                max_new_tokens,

            ],

            outputs=[

                chatbot,

                conversation_state,

                message,

                metrics,

            ],
        )


        # ====================================================
        # Clear Button
        # ====================================================

        clear_button.click(

            fn=clear_chat,

            inputs=[],

            outputs=[

                chatbot,

                conversation_state,

                message,

                metrics,

            ],
        )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo.queue()

    demo.launch(

        server_name=(
            "127.0.0.1"
        ),

        server_port=7860,

        show_error=True,

        inbrowser=True,

        theme=theme,

        css=CSS,
    )