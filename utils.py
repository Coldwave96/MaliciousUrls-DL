def read_data(file_path):
    with open(file_path) as files:
        urls = []
        labels = []
        for line in files.readlines():
            temp = line.split('\t')
