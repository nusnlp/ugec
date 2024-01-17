def write_txt(data, path):
    with open(path, 'w', encoding='utf-8')as f:
        for d in data:
            f.write(f'{d}\n')


def read_txt(path):
    with open(path, 'r', encoding='utf-8')as f:
        lines = [l.strip('\n') for l in f.readlines()]
        return lines
