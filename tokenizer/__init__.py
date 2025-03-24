def process_special_pinyin(pinyins):
    res = []
    for pinyin in pinyins:
        if pinyin[-1] == "r":
            res = res + [pinyin[:-1], "er"]
        else:
            res = res + [pinyin]
    return res