import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('savemodel/model_s_v1_trace.pt', map_location="cpu")
# model = model.to(DEVICE)
model.eval()
image_size = (224, 224)

if __name__ == '__main__':
    input_x = torch.rand(1, 1, image_size[1], image_size[0])

    torch.onnx.export(model,
                        input_x,
                        input_names=["input"],
                        output_names=["output"],
                        f="savemodel/model.onnx")