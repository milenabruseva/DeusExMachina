class QFeatures:
    __slots__ = ("posx", "posy")
    posx : int # an example feature
    posy: int  # an example feature

    def __init__(self, posx: int, posy: int):
        self.posx = posx
        self.posy = posy

    def __repr__(self):
        return str(self.posx) +"|"+ str(self.posy)