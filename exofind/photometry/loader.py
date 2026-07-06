from lightkurve import read


def load_tpf(path):
    """
    Reads a Target Pixel File (.fits)
    """
    return read(path)

def load_lightcurve(path):
    """
    Load a TESS Light Curve (.fits)
    """
    return read(path)