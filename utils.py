def load_data(malicious_path, benign_path):
    urls = []
    labels = []

    with open(malicious_path) as f:
        for line in f.readlines():
            urls.append(line)
            labels.append(1)

    with open(benign_path) as f:
        for line in f.readlines():
            urls.append(line)
            labels.append(0)

    return urls, labels


def split_url(url, part):
    if url.startswith("https://"):
        url = url[8:]
    if url.startswith("http://"):
        url = url[7:]
    if url.startswith("www."):
        url = url[4:]

    slash_index = url.find('/')

    if 0 < slash_index < len(url) - 1:  # url = "example.com/example/example.conf"
        domain = url[:slash_index]
        args = url[slash_index+1:]
        tokens = args.split('/')
        token = "/".join(tokens[:-1])
        last_token = tokens[-1]
        if len(tokens) > 2 and last_token == '':
            token = "/".join(tokens[:-2])
            last_token = tokens[-2]
        question_index = last_token.find('?')
        if question_index != -1:
            args = last_token[question_index+1:]
            token = token + "/" + last_token[:question_index]
        else:
            args = ''
            token = token + "/" + last_token
        last_slash_index = token.rfind('/')
        sub_dir = token[:last_slash_index]
        file_name = token[last_slash_index+1:]
        file_last_dot_index = file_name.rfind('.')
        if file_last_dot_index != -1:
            file_type = file_name[file_last_dot_index+1:]
            file_name = file_name[:file_last_dot_index]
        else:
            file_type = ''
    elif slash_index == 0:   # url = "/example"
        domain = url[1:]
        token = ''
        args = ''
        sub_dir = ''
        file_name = ''
        file_type = ''
    elif slash_index == len(url) - 1:  # url = "example.com/"
        domain = url[:-1]
        token = ''
        args = ''
        sub_dir = ''
        file_name = ''
        file_type = ''
    else:  # url = "example.com"
        domain = url
        token = ''
        args = ''
        sub_dir = ''
        file_name = ''
        file_type = ''

    if part == "domain":
        return domain
    elif part == "path":
        return token
    elif part == "args":
        return args
    elif part == "sub_dir":
        return sub_dir
    elif part == "file_name":
        return file_name
    elif part == "file_type":
        return file_type
    elif part == "others":
        if len(args) > 0:
            return token + '?' + args
        else:
            return token
    else:
        return domain, token, args, sub_dir, file_name, file_type

