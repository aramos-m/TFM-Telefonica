def formato_srt(segundos):
    """
    Convierte segundos en formato SRT (HH:MM:SS,mmm)
    """
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    seg = int(segundos % 60)
    milisegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02}:{minutos:02}:{seg:02},{milisegundos:03}"
