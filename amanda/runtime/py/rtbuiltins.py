def lista(ty, elements):
    return elements


def converta(obj, ty):
    if ty in (int, float, str, bool):
        return ty(obj)
    else:
        return obj
