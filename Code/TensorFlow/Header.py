with open('export/RoadLiteMobileNetV2.tflite', 'rb') as f:
    data = f.read()

with open('export/model_data.h', 'w') as f:
    f.write('const unsigned char model_data[] = {\n')
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write('\n ')
        f.write(f' 0x{b:02x},')
    f.write('\n};\nconst int model_data_len = sizeof(model_data);\n')
