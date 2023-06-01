import gradio as gr
import os
import torch

from model import create_effnetb2
from timeit import default_timer as timer


def main():
    # # Device agnostic
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # # Setup class names
    class_names = ["pizza", "steak", "sushi"]

    # # Create model
    effnetb2, effnetb2_transforms = create_effnetb2(
        out_features=len(class_names),
        device=device)

    effnetb2.load_state_dict(torch.load(
        f="effnetb2.pth",
        map_location=torch.device(device)))

    def predict(img):
        """
        Transforms and performs a prediction on img
        Returns prediction and time taken.
        """
        start_time = timer()

        transformed_img = effnetb2_transforms(img).unsqueeze(0).to(device)

        effnetb2.to(device)
        effnetb2.eval()

        with torch.inference_mode():
            pred_logit = effnetb2(transformed_img)
            pred_prob = torch.softmax(input=pred_logit, dim=1)

        pred_labels_and_probs = {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))} # noqa 5501

        pred_time = round(timer() - start_time, 5)

        return pred_labels_and_probs, pred_time

    # # Gradio app
    title = "FoodVision Mini 🍕🥩🍣"
    description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi." # noqa 5501
    article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)." # noqa 5501
    example_list = [["examples/" + example] for example in os.listdir("examples")] # noqa 5501

    # Create demo
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                 gr.Number(label="Prediction time (s)")],
        examples=example_list,
        title=title,
        description=description,
        article=article)

    demo.launch()


if __name__ == "__main__":
    main()
