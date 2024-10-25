from ultralytics import FastSAM

# Create a FastSAM model
model = FastSAM('FastSAM-s.pt')

# Run interfaces on the source
results = model(source=0, show=True, conf=0.4, save=True)