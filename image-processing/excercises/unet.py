import segmentation_models_pytorch as smp

in_channels = 3 
out_channels = 1

model = smp.Unet(encoder_name="resnet34", in_channels=in_channels, classes=out_channels, encoder_weights="imagenet")

def model():
    return model