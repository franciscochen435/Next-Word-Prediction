from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("trained_tokenizer/tokenizer.json")  # path to your saved tokenizer

test_sentence = "unbelievably, the tokenization algorithm merges substrings intelligently."
encoded = tokenizer.encode(test_sentence)

print("Input sentence:")
print(test_sentence)

print("\nTokens:")
print(encoded.tokens)

print("\nToken IDs:")
print(encoded.ids)