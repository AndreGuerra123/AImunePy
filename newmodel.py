from keras.applications import mobilenet

model = mobilenet.MobileNet()

# Save the model architecture
with open('mobilenet.json', 'w') as f:
    f.write(model.to_json())