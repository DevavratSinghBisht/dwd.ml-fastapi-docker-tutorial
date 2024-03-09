import io
import torch
from model_store.cifar10_cnn_v1 import load_model, infer_transform, OP_TO_LABELS


device = ("cuda:0" if torch.cuda.is_available() else 'cpu')

model = load_model(device=device)

async def get_predictions(file_content: bytes, gpu: bool = True):

    file_content = io.BytesIO(file_content)

    image = infer_transform(file_content)
    image = image.reshape(1, *image.shape)
    # print("image device", image.to(device).device)
    # print("model device", model.device)
    preds = model(image.to(device))
    predicted_class = OP_TO_LABELS[preds.max(1)[1].cpu().numpy().tolist()[0]]

    return predicted_class