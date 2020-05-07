NETWORK = {
    "inception": [
        {'name': 'recog/InceptionResnetV1/Repeat/block35_1/add:0', 'shape': [None, 17, 17, 256]},  # 0 A-1
        {'name': 'recog/InceptionResnetV1/Repeat/block35_3/add:0', 'shape': [None, 17, 17, 256]},  # 1 A-3
        {'name': 'recog/InceptionResnetV1/Repeat/block35_4/add:0', 'shape': [None, 17, 17, 256]},  # 2 A-4
        {'name': 'recog/InceptionResnetV1/Repeat/block35_5/add:0', 'shape': [None, 17, 17, 256]},  # 3 A-5
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_1/add:0', 'shape': [None, 8, 8, 896]},  # 4 B-1
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_2/add:0', 'shape': [None, 8, 8, 896]},  # 5 B-2
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_3/add:0', 'shape': [None, 8, 8, 896]},  # 6 B-3
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_4/add:0', 'shape': [None, 8, 8, 896]},  # 7 B-4
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_5/add:0', 'shape': [None, 8, 8, 896]},  # 8 B-5
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_6/add:0', 'shape': [None, 8, 8, 896]},  # 9 B-6
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_7/add:0', 'shape': [None, 8, 8, 896]},  # 10 B-7
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_8/add:0', 'shape': [None, 8, 8, 896]},  # 11 B-8
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_9/add:0', 'shape': [None, 8, 8, 896]},  # 12 B-9
        {'name': 'recog/InceptionResnetV1/Repeat_1/block17_10/add:0', 'shape': [None, 8, 8, 896]},  # 13 B-10
        {'name': 'recog/InceptionResnetV1/Repeat_2/block8_1/add:0', 'shape': [None, 3, 3, 1792]},  # 14 C-1
        {'name': 'recog/InceptionResnetV1/Repeat_2/block8_2/add:0', 'shape': [None, 3, 3, 1792]},  # 15 C-2
        {'name': 'recog/InceptionResnetV1/Repeat_2/block8_3/add:0', 'shape': [None, 3, 3, 1792]},  # 16 C-3
        {'name': 'recog/InceptionResnetV1/Repeat_2/block8_4/add:0', 'shape': [None, 3, 3, 1792]},  # 17 C-4
        {'name': 'recog/InceptionResnetV1/Repeat_2/block8_5/add:0', 'shape': [None, 3, 3, 1792]},  # 18 C-5
        {'name': 'recog/InceptionResnetV1/Block8/add:0', 'shape': [None, 3, 3, 1792]}  # 19
    ]
}


def get_placeholder(name, index):
    if name not in NETWORK: raise Exception("Network name does not exist!")
    if index < 0 or index >= len(NETWORK[name]): raise Exception("Network layer out of range")
    return NETWORK[name][index]
