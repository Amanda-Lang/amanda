def e_alg(_r10_):
    return embutidos_ama.txt_contem("0123456789", _r10_)


def e_operador(_r10_):
    return embutidos_ama.txt_contem("+-*/", _r10_)


def anexa_txt(_r10_, _r11_):
    return f"{_r10_}{_r11_}"


def processar_cmd(_r10_):
    _r11_ = "Comando mal formatado. Comandos devem estar na forma: {operando1} {operador} {operando2}"
    _r12_ = ""
    _r13_ = ""
    _r14_ = ""
    _r15_ = 0
    _r16_ = _r10_[_r15_]
    _r17_ = ""
    _r18_ = embutidos_ama.tam(_r10_)
    while e_alg(_r10_[_r15_]) and (_r15_ < _r18_):
        _r16_ = _r10_[_r15_]
        _r12_ = anexa_txt(_r12_, _r16_)
        _r15_ = _r15_ + 1

    _r16_ = _r10_[_r15_]
    if (_r15_ >= _r18_) or (not e_operador(_r16_)):
        embutidos_ama.escrevaln(_r11_)
        return None

    _r13_ = _r16_
    _r15_ = _r15_ + 1
    while _r15_ < _r18_:
        _r16_ = _r10_[_r15_]
        _r14_ = anexa_txt(_r14_, _r16_)
        _r15_ = _r15_ + 1

    _r19_ = _re0_
    _re0_


def inicio():
    _r10_ = ""
    while verdadeiro:
        _r10_ = embutidos_ama.leia("> ")
        if _r10_ == "fim":
            break

        processar_cmd(_r10_)


inicio()
