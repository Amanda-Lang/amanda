# builtin types
int = int
texto = str
real = float
bool = bool


def escreva(obj):
    print(obj, end="")


def escrevaln(obj):
    print(obj, end="\n")


def leia(prompt: str):
    return input("> ")


def tam(obj):
    return len(obj)


def txt_contem(haystack: str, needle: str):
    return needle in haystack
