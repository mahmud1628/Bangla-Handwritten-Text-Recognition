"""Launcher for the modular Bengali HTR full pipeline app."""

import gradio as gr

from htr_pipeline.ui import create_interface


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Bengali HTR Full Pipeline GUI")
    print("=" * 70 + "\n")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )