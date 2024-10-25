from ultralytics import SAM

#Load a model
model = SAM('sam_b.pt')

# Display model information(optional)
# model.info()

results = model(source=0,show=True,conf=0.4,save=True)