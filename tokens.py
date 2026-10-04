
from collections import Counter
import torch
with open ("training.txt","r",encoding="utf-8") as f:
    text = f.read()

characters = [list(text)]
def find_pair(characters):
    pair_count = Counter()
    for i in characters:
        for j in range(len(i) - 1):
            pair = (i[j],i[j+1])
            pair_count[pair]+=1
    if not pair_count:
        return None
    return max(pair_count,key=pair_count.get)

def merge_pairs(characters,best_pair):
    merged = []
    left,right = best_pair[0],best_pair[1]
    for i in characters:
        j = 0
        while j < len(i) - 1:
            if j < len(i) - 1 and i[j] == left and i[j+1] == right:
                merged.append(i[j]+i[j+1])
                j+=2
            else:
                merged.append(i[j])
                j+=1
    return merged

def train_token(characters):
    merged_rules = []
    for _ in range(200):
        best_pair = find_pair(characters)
        if best_pair is None:
            return "Not pair found end"
        merged_values = merge_pairs(characters,best_pair)
        merged_rules.append([best_pair[0],best_pair[1]])
        characters = [merged_values]
        print(best_pair)
    return merged_rules,merged_values

merged_rules,merged_values = train_token(characters)
vocab = sorted(set(col for row in merged_values for col in row))
vocab = ["<pad>","<eos>"] + vocab
token_id = {j:i for i,j in enumerate(vocab)}
id_token = {j:i for i,j in token_id.items()}

encoded = [token_id[i] for k in merged_values for i in k]

print(len(vocab))
print(len(token_id))
print(len(id_token))
print(len(encoded))

torch.save({
    "token_id":token_id,
    "id_token":id_token,
    "vocab":vocab,
    "encoded":encoded
},"token.pt")

