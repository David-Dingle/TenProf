from torchvision import transforms
from PIL import Image
import urllib
import time
import torch
import sys

device = str(sys.argv[1])
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.eval()
# Download an example image from the pytorch website
url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# move the input and model to GPU for speed if available
if torch.cuda.is_available() and device == 'cuda':
    input_batch = input_batch.to(device)
    model.to(device)

start = time.time()
output = model(input_batch)
end = time.time()

print('{:.2f}s'.format(end - start))
