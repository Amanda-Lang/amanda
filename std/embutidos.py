# builtin types
int = int
texto = str
real = float
bool = bool


def _fmt_object(obj):
    match obj:
        case bool():
            return "verdadeiro" if obj else "falso"
        case _:
            return obj


def escreva(obj):
    print(_fmt_object(obj), end="")


def escrevaln(obj):
    print(_fmt_object(obj), end="\n")


def leia(prompt: str):
    return input("> ")


def tam(obj):
    return len(obj)


def txt_contem(haystack: str, needle: str):
    return needle in haystack
