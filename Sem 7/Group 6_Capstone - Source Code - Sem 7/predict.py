import argparse
import pickle

import torch
from PIL import Image
from torchvision import models, transforms


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(image_path, model_path):
    with open(model_path, "rb") as handle:
        package = pickle.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    backbone = models.densenet121(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    features = backbone(tensor).cpu().numpy()

    classifier = package["classifier"]
    predicted_index = int(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    class_names = package["class_names"]
    confidence = float(probabilities[predicted_index])
    print(f"Prediction: {class_names[predicted_index]}")
    print(f"Confidence: {confidence:.4f}")
    for name, probability in zip(class_names, probabilities):
        print(f"{name}: {probability:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    predict(args.image, args.model)


if __name__ == "__main__":
    main()
