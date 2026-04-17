import gradio as gr

from .pipeline import process_full_document


def create_interface():
    with gr.Blocks(title="Bengali HTR - Full Pipeline") as demo:
        gr.Markdown(
            """
            # Bengali Handwritten Text Recognition - Full Pipeline
            ### Complete System: Image Segmentation -> Text Recognition

            Upload an image of Bengali handwritten text to get the full transcription.
            The system will automatically:
            1. Detect and segment lines
            2. Detect and segment words
            3. Recognize text from each word
            4. Combine into full document text
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Handwritten Document Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400,
                )

                num_beams_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Recognition Quality (Beam Search Width)",
                    info="Higher = better quality but slower (recommended: 3)",
                )

                process_btn = gr.Button("Process Document", variant="primary", size="lg")

            with gr.Column():
                preprocessed_preview = gr.Image(
                    label="Preprocessed Image (Used for Pipeline)",
                    type="numpy",
                    height=220,
                )

                output_text = gr.Textbox(
                    label="Recognized Text (Full Document)",
                    lines=10,
                    placeholder="The recognized text will appear here...",
                )

                detailed_output = gr.Textbox(
                    label="Detailed Output (Line by Line)",
                    lines=8,
                    placeholder="Detailed line-by-line output with statistics...",
                )

        gr.Markdown(
            """
            ### Tips for Best Results:
            - Use clear, well-lit images with good contrast
            - Ensure text is horizontal (not too skewed)
            - Higher quality images gives better recognition
            - Works with full pages, paragraphs, or sentences
            - Processing time is usually 10 to 30 seconds based on image size

            ### Supported:
            - Image formats: JPG, PNG, JPEG
            - Content: Bengali handwritten text (lines, paragraphs, full pages)
            """
        )

        process_btn.click(
            fn=process_full_document,
            inputs=[image_input, num_beams_slider],
            outputs=[output_text, detailed_output, preprocessed_preview],
        )

    return demo
