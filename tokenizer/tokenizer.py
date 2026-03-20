from utils import read_data

class Tokenizer:
    def __init__(self, size, special_tokens = None, end_of_word = "<w>"):
        self.size = size
        self.special_tokens = special_tokens or []
        self.end_of_word = end_of_word

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
                
def main():
    bpe = Tokenizer(size = 32000, special_tokens = None, end_of_word = "</w>")

    tokens = []
    for line in read_data("wiki.train.txt"):
        words = line.strip().split()
        tokens.extend(words)

    vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
    ids = [vocab[token] for token in tokens]

    vocab_size = 32000
    num_merges = vocab_size - 256
    ids = list(tokens)

    merges = {}
    for i in range(num_merges):
        stats = bpe.get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = bpe.merge(ids, pair, idx)
        merges[pair] = idx

    # for i in range(num_merges):
    #     stats = bpe.get_stats(ids)
    #     if not stats:
    #         print("No more pairs to merge.")
    #         break
    #     pair = max(stats, key=stats.get)

    #     new_idx = len(vocab)

    #     print(f"Merging {pair} into new token ID {new_idx}")
    #     ids = bpe.merge(ids, pair, new_idx)
    #     merges[pair] = new_idx
    #     merged_token_str = "".join([str(pair[0]), str(pair[1])])
    #     vocab[merged_token_str] = new_idx

if __name__ == "__main__":
    main()