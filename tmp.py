
def representative_dataset(set):
    for sample in set:
        yield [sample]

abc = [5, 4, 3, 2, 1]
for _ in range(4):
    lam = lambda x=abc: representative_dataset(x)
    for val in lam():
        print(val)
