import random
import hashlib

class TNC_ECC_Module:
    def __init__(self, G, order):
        self.G = G
        self.order = order
        self.pk = None
        self.sk = None

    def KeyGen(self): # pg 7: https://doi.org/10.3390/sym13081330
        a = random.randint(1, self.order - 1)
        y_1 = self.G * a
        y_2 = self.G * a * a
        self.pk = (y_1, y_2)
        self.sk = a
        keys = (self.pk, self.sk)
        return keys

    def Sign(self, m): # m must be a string (not bytes)
        r = random.randint(1, self.order - 1)
        U = self.G * r
        V = self.pk[0] * r
        x = self.Selected_H(m, U, V)
        s = r + x*self.sk
        sigma = (s, x)
        return sigma

    def Verify(self, m, sigma):
        U_prime = self.G * sigma[0] + self.pk[0] * (-sigma[1])
        V_prime = self.pk[0] * sigma[0] + self.pk[1] * (-sigma[1])
        x_prime = self.Selected_H(m, U_prime, V_prime)
        return x_prime == sigma[1]

    # TNC-IBI, Kurosawa-Heng Transform.
    def MKGen(self):
        keys = self.KeyGen()
        return keys

    def UKGen(self, ID):
        uk = self.Sign(ID)
        return uk

    def Selected_H(self, x, G1, G2): # {0, 1}* x G x G -> Z_q
        to_hash = (str(x),
                   str(G1.x()),
                   str(G1.y()),
                   str(G2.x()),
                   str(G2.y()))
        hash_to_encode = "".join(to_hash)
        h = hashlib.sha256(hash_to_encode.encode()).digest()
        return int.from_bytes(h, 'big') % self.order