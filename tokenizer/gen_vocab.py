train_file = "./dataset/split/train/pinyin"
dev_file = "./dataset/split/dev/pinyin"
test_file = "./dataset/split/test/pinyin"
vocab_path = "./tokenizer/vocab.txt"

res = set()
for file in [train_file, dev_file, test_file]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pinyins = parts[1]
                pinyin_list = pinyins.split(" ")

                for pinyin in pinyin_list:
                    if pinyin != "":
                        res.add(pinyin)

res = list(res)


with open(vocab_path, "w", encoding="utf-8") as f:
    for pinyin in res:
        f.write(pinyin + "\n")
