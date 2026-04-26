import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import get_resNet18
from preprocessing import get_resNet18_transforms

def generate_gradcam(model_path, image_path, save_path, transforms, idx_to_class, num_classes, device):
    model = get_resNet18(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    target_layers = [model.layer4[-1]]

    raw_img = Image.open(image_path).convert("RGB")
    in_tensor = transforms(raw_img).unsqueeze(0).to(device)
    rgb_img = np.array(raw_img.resize((224, 224))) / 255.0

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )
    
    outputs = model(in_tensor)
    pred_class = outputs.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(
        input_tensor=in_tensor,
        targets=targets
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    true_label = Path(image_path).parent.name
    ax[0].imshow(rgb_img)
    ax[0].set_title(f"Original image of {true_label}")
    ax[0].axis("off")

    ax[1].imshow(rgb_img)
    ax[1].imshow(
        grayscale_cam,
        cmap="jet",
        alpha=grayscale_cam * 0.7   
    )
    ax[1].set_title(f"Grad-CAM: predicted class {idx_to_class[pred_class]}")
    ax[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, f"{true_label}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        
test_dir = r"data\Test"
classes = sorted(os.listdir(test_dir))
idx_to_class = {
    i: cls_name
    for i, cls_name in enumerate(classes)
}  

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"results\resnet18\resnet18_weighted_focal.pth"

img_path = r"data\Test\squamous cell carcinoma\ISIC_0012079.jpg"
save_path = r"results\resnet18\explainability"
_, transforms = get_resNet18_transforms()
generate_gradcam(model_path, img_path, save_path, transforms, idx_to_class, 9, device)
