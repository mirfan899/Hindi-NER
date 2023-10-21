from sklearn.model_selection import train_test_split


sentences_list = []
tags_list = []
tags = []
sentence = []

path = "Datasets/Hindi/hindi.txt"
with open(path, encoding="utf-8-sig") as reader:
    for line in reader:
        if line == "\n":
            sentences_list.append(sentence)
            sentence = []
            tags_list.append(tags)
            tags = []
        else:
            word, tag = line.strip().split("\t")
            sentence.append(word)
            tags.append(tag)


X_train, X_test, y_train, y_test = train_test_split(sentences_list, tags_list, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

with open("Datasets/Hindi/hindi_train.txt", "w") as writer:
    for x, y in zip(X_train, y_train):
        for word, label in zip(x, y):
            writer.write(f"{word}\t{label}\n")
        writer.write(f"	\n")

with open("Datasets/Hindi/hindi_test.txt", "w") as writer:
    for x, y in zip(X_test, y_test):
        for word, label in zip(x, y):
            writer.write(f"{word}\t{label}\n")
        writer.write(f"	\n")

with open("Datasets/Hindi/hindi_validation.txt", "w") as writer:
    for x, y in zip(X_val, y_val):
        for word, label in zip(x, y):
            writer.write(f"{word}\t{label}\n")
        writer.write(f"	\n")

print("Done splitting ....")
