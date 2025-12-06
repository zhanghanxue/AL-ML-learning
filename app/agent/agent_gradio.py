import gradio as gr
import requests
import os
from typing import Dict, Any


def get_base_url():
    space_id = os.getenv("SPACE_ID")
    if space_id:
        author, space_name = space_id.split("/")
        return f"https://{author}-{space_name}.hf.space"
    return "http://localhost:8000"


def process_with_agent(ticket_text: str) -> Dict[str, Any]:
    try:
        base_url = get_base_url()
        response = requests.post(
            f"{base_url}/agent/predict",
            json={"text": ticket_text},
            timeout=45,
            verify=False,  # For local testing
        )

        if response.status_code == 200:
            result = response.json()

            response_text = result.get("response", "No response")
            intent = result.get("intent", "unknown")
            confidence_val = f"{result.get('confidence', 0):.1%}"
            priority_val = "🔴 URGENT" if result.get("is_urgent") else "🟢 Normal"
            escalated_val = "✅ Yes" if result.get("escalated") else "❌ No"
            processing_time_val = f"{result.get('processing_time', 0):.2f}s"
            cost_val = f"${result.get('llm_cost', 0):.6f}"
            
            # Return as tuple of individual values + raw result
            return (
                response_text,
                intent,
                confidence_val,
                priority_val,
                escalated_val,
                processing_time_val,
                cost_val,
                result,  # Raw response for JSON component
            )
        else:
            # Return error values
            return (
                f"Error: API returned {response.status_code}",
                "error",
                "0%",
                "Unknown",
                "No",
                "0s",
                "$0",
                {"error": f"HTTP {response.status_code}"},
            )

    except Exception as e:
        return {
            "response": f"Connection error: {str(e)}",
            "intent": "error",
            "confidence": "0%",
            "priority": "Unknown",
            "escalated": "No",
            "processing_time": "0s",
            "cost": "$0",
        }, {"error": str(e)}


def create_agent_interface():
    """Create a dedicated interface for the agent"""
    with gr.Blocks(title="🤖 Intelligent Support Agent") as demo:
        gr.Markdown("# 🤖 Intelligent Support Ticket Agent")
        gr.Markdown(
            """
        **Week 12 Feature**: AI Agent with Reasoning
        
        This agent doesn't just classify - it:
        - 🔍 **Understands** your ticket
        - 🤔 **Reasons** about what to do
        - 🚨 **Acts** on urgent issues
        - 💰 **Tracks** costs
        
        Compare with the simple classifier on the other tab!
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                ticket_input = gr.Textbox(
                    label="Support Ticket",
                    placeholder="Describe your issue...",
                    lines=5,
                )

                with gr.Row():
                    submit_btn = gr.Button("🧠 Process with Agent", variant="primary")

            with gr.Column(scale=1):
                # Output fields
                agent_response = gr.Textbox(label="Agent Response", interactive=False)
                predicted_intent = gr.Textbox(label="Predicted Intent", interactive=False)
                confidence = gr.Textbox(label="Confidence", interactive=False)
                priority = gr.Textbox(label="Priority", interactive=False)
                escalated = gr.Textbox(label="Escalated to Slack", interactive=False)
                processing_time = gr.Textbox(label="Processing Time", interactive=False)
                cost = gr.Textbox(label="LLM API Cost", interactive=False)

        # Example tickets that show off the agent
        examples = [
            "My credit card was stolen and someone made unauthorized purchases!",
            "I need to check my account balance for last month.",
            "My card is not working at any ATM or store.",
            "I suspect my account has been hacked - strange transactions!",
            "URGENT: I lost my debit card at the airport!",
        ]

        gr.Examples(examples=examples, inputs=ticket_input, label="Try these examples:")

        # Connect button
        submit_btn.click(
            fn=process_with_agent,
            inputs=ticket_input,
            outputs=[
                agent_response,
                predicted_intent,
                confidence,
                priority,
                escalated,
                processing_time,
                cost,
                gr.JSON(label="Raw Response"),  # Keep JSON for raw data if available
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_agent_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861)
