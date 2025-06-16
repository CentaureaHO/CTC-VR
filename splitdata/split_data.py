
import os


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_chinese_data(file_path):
    res = {}

    i = 0
    id = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if i % 2 == 0:
                id_parts = line.split('\t', 1)
                if len(id_parts) == 2:
                    id = id_parts[0]
                    text = id_parts[1]
                    res[id] = {"text": text}
                else:
                    raise ValueError(f"Invalid line format: {line}")
            else:
                res[id]["pinyin"] = line
            i = i+1

    return res


def process_special_pinyin(pinyins):
    pinyins = pinyins.split(" ")
    res = []
    for pinyin in pinyins:
        if pinyin[-1] in "1234567890":
            tone = pinyin[-1]
            pinyin = pinyin[:-1]
        else:
            tone = ""
        if pinyin[-1] == "r":
            res = res + [pinyin[:-1] + tone, "er"]
        elif pinyin != "IY" and pinyin != "P":
            res = res + [pinyin + tone]
        elif pinyin == "P":
            res = res + ["pi"]
    t = ""
    for i in range(len(res)):
        t += res[i]
        if i != len(res) - 1:
            t += " "
    return t


def write_data(data, dilename):
    create_directory(dilename)

    with open(f"{dilename}/wav.scp", 'w', encoding="utf-8") as f:
        for id in data:
            f.write(id + '\t' + f"Wave/{id}.wav" + "\n")
    with open(f"{dilename}/pinyin", 'w', encoding="utf-8") as f:
        for id in data:
            pinyin = process_special_pinyin(data[id]['pinyin'])
            pinyin = pinyin.split(" ")
            res = ""
            for i in range(len(pinyin)):
                sub = pinyin[i]
                if sub[-1] in "1234567890":
                    sub = sub[:-1]
                res += sub
                if i != len(pinyin) - 1:
                    res += " "
            f.write(id + '\t' + res+"\n")


def split_dataset(dir="./dataset/split/", filename="./dataset/ProsodyLabeling/000001-010000.txt"):
    res = parse_chinese_data(filename)

    train = {}
    test = {}
    dev = {}

    for key in res:
        num = int(key)
        if num <= 8000:
            train[key] = res[key]
        elif num <= 9000 and num > 8000:
            dev[key] = res[key]
        else:
            test[key] = res[key]

    print(len(train), len(dev), len(test))

    create_directory(dir)

    write_data(train, f"{dir}/train")
    write_data(dev, f"{dir}/dev")
    write_data(test, f"{dir}/test")


if __name__ == "__main__":
    split_dataset()
