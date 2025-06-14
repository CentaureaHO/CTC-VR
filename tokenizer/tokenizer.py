class Tokenizer:
    def __init__(self, vocab_file = "./tokenizer/vocab.txt"):
        self.token2id = {}
        self.id2token = {}
        count = 0

        self.special_token = ["<pad>", "<unk>", "<sos>", "<eos>", " ", "<blk>"]
        for token in self.special_token:
            self.token2id[token] = count
            self.id2token[count] = token
            count += 1

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                # 去掉空格和换行
                line = line.strip()
                self.token2id[line] = count
                self.id2token[count] = line
                count += 1

    def __call__(self, s:list):
        """
        输入token list
        返回 token id list
        """
        token_ids = []
        for token in s:
            token_ids.append(self.token2id.get(token, self.token2id["<unk>"]))
        return token_ids

    def decode(self, ids: list, ignore_special=True):
        """
        输入为 token id list
        返回为token list
        ignore_special: 是否忽略特殊字符
        """         
        tokens = []
        for id_val in ids:
            token = self.id2token.get(id_val)
            if token is None:
                if not ignore_special:
                    tokens.append("<unk>")
                continue

            if ignore_special and token in self.special_token:
                continue
            tokens.append(token)
        return tokens
    
    def special_token_ids(self):
        return [self.token2id[token] for token in self.special_token]
    
    def size(self):
        return len(self.token2id)
    
    def padding_id(self):
        return self.token2id["<pad>"]
    
    def blk_id(self):
        return self.token2id["<blk>"]
