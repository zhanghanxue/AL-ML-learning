import gradio as gr
import requests
import os


# Get the base URL for the API
def get_base_url():
    """Get the correct base URL for API calls"""
    # In Hugging Face Space, use relative URL since it's the same container
    if os.getenv("SPACE_ID"):  # Hugging Face sets this environment variable
        return ""  # Relative URL works in same container
    else:
        return "http://localhost:8000"  # For local development


def classify_ticket(ticket_text):
    """Send ticket text to the API and get classification."""
    try:
        base_url = get_base_url()
        response = requests.post(
            f"{base_url}/predict",
            json={"text": ticket_text},
            timeout=30,
            verify=False,  # Disable SSL verification for local testing
        )

        if response.status_code == 200:
            result = response.json()
            intent = result.get("prediction", "unknown")
            confidence = f"{result.get('confidence', 0):.2f}"
            return intent, confidence, result
        else:
            error_msg = f"API error: {response.status_code}"
            return "Error", "0.00", {"error": error_msg}

    except Exception as e:
        error_msg = f"Connection failed: {str(e)}"
        return "Error", "0.00", {"error": error_msg}


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Support Ticket Triage Agent") as demo:
        gr.Markdown("# 🎫 Support Ticket Triage Agent")
        gr.Markdown(
            "Enter a support ticket below to automatically classify its intent."
        )

        with gr.Row():
            with gr.Column():
                ticket_input = gr.Textbox(
                    label="Support Ticket Text",
                    placeholder="Enter the support ticket description here...",
                    lines=4,
                    max_length=10,
                )
                classify_button = gr.Button("Classify Ticket", variant="primary")

            with gr.Column():
                intent_output = gr.Label(label="Predicted Intent")
                confidence_output = gr.Textbox(
                    label="Confidence Score", interactive=False
                )
                raw_output = gr.JSON(label="Raw API Response")

        # Example tickets for quick testing
        gr.Examples(
            examples=[
                "I need help resetting my password.",
                "How can I update my billing information?",
                "My internet connection is slow and keeps dropping.",
                "I want to cancel my subscription.",
                "Where can I find the user manual for my device?",
            ],
            inputs=ticket_input,
        )

        # Connect button to function
        classify_button.click(
            fn=classify_ticket,
            inputs=ticket_input,
            outputs=[intent_output, confidence_output, raw_output],
        )

    return demo


# For running Gradio standalone (if needed)
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
