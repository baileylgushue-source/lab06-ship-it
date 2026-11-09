"""A Gradio page that asks about an order and says whether to offer a voucher."""

import gradio as gr

from predict import (
    CHOICES,
    INTEGER_FIELDS,
    MODEL_INFO,
    NUMERIC_LIMITS,
    predict_order,
)

try:  # present only on Hugging Face ZeroGPU hardware
    import spaces
except ImportError:
    spaces = None
if spaces is not None:

    @spaces.GPU
    def keep_zerogpu_happy() -> None:
        """Does nothing. ZeroGPU only needs GPU function to start."""


FEATURES = MODEL_INFO["numeric_features"] + MODEL_INFO["categorical_features"]

LABELS = {
    "distance_km": "Distance (km)",
    "items": "Items in the order",
    "hour": "Hour order placed",
    "prep_minutes": "Kitchen prep minutes (optional)",
    "courier_months": "Courier months with QuickBites",
    "weather": "Weather",
    "day_type": "Day type",
    "vehicle": "Vehicle",
}


def build_input(name):
    """Make the right Gradio component for one column of the model."""
    label = LABELS.get(name, name)
    if name in CHOICES:
        choices = sorted(CHOICES[name])
        return gr.Dropdown(choices, value=choices[0], label=label)
    low, high = NUMERIC_LIMITS[name]
    if name in INTEGER_FIELDS:
        return gr.Slider(low, high, step=1, value=low, label=label)
    note = f"Allowed: {low} to {high}"
    if name == "prep_minutes":
        note += ", or leave empty"
    return gr.Number(label=label, info=note)


def score(*values):
    """Turn the page values into an order, score it, and describe the result."""
    order = dict(zip(FEATURES, values, strict=True))

    try:
        result = predict_order(order)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    for warning in result["warnings"]:
        gr.Warning(warning)

    chances = {
        "Will arrive late": result["probability"],
        "Will arrive on time": result["probability_on_time"],
    }
    action = f"{result['message']} (threshold used: {result['threshold']:.2f})"
    return chances, action


with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown("# Quickbite Late Order Check\nWill this order arrive late?")
    with gr.Row():
        with gr.Column():
            inputs = [build_input(name) for name in FEATURES]
            button = gr.Button("Check order", variant="primary")
        with gr.Column():
            chances_out = gr.Label(num_top_classes=2, label="Chance of each outcome")
            action_out = gr.Textbox(label="What to do")
    button.click(score, inputs=inputs, outputs=[chances_out, action_out])

if __name__ == "__main__":
    demo.launch()
