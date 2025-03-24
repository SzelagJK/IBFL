import numpy as np
import hashlib
from Crypto.Util.number import getPrime


def sample_matrix(rows, cols, bound):
    return np.random.randint(-bound, bound + 1, size=(rows, cols))

def sample_vector(dim, bound):
    return np.random.randint(-bound, bound + 1, size=(dim, 1))

class LB_IBI_Module:
    def __init__(self, q):
        # pg. 10, Table 1: https://icics2024.aegean.gr/wp-content/uploads/2024/08/150570198.pdf
        # Using security level at 90 bits
        self.q = q
        self.n = 256
        self.m = 5
        self.k = 7
        self.eta = 1
        self.gamma1 = 49920
        self.beta1 = 39
        self.gamma2 = 4984012800
        self.beta2 = 1946880
        self.kappa = 39
        self.mpk = None
        self.msk = None

    def mod_q(self, x): # apply mod to all elements of a matrix
        return np.mod(x, self.q)

    # --------------------------- LB-IBS ALGORITHMS ---------------------------
    # Note: based on average-case hardness of the SIS problem
    # pg. 6: https://icics2024.aegean.gr/wp-content/uploads/2024/08/150570198.pdf
    def Setup(self):
        A = np.random.randint(0, self.q, size=(self.n, self.m))  # A in ℤ_q
        S = sample_matrix(self.m, self.k, self.eta)  # S with small coefficients
        T = self.mod_q(np.dot(A, S))
        self.mpk = {"A": A, "T": T}
        self.msk = {"S": S}
        key_pair = (self.mpk, self.msk)
        return key_pair


    def Extract(self, mpk, msk, identity):
        A = mpk["A"]
        S = msk["S"]
        while True:
            Y = sample_matrix(self.m, self.k, self.gamma1)
            W = self.mod_q(np.dot(A, Y))
            C = self.H1(W, identity, self.k, self.kappa)
            Z = np.dot(S, C) + Y  # no mod reduction
            if np.all(np.abs(Z) < (self.gamma1 - self.beta1)):
                break
        skid = {"S_id": Z, "W": W}
        return skid


    def Sign(self, mpk, skid, message, identity):
        A = mpk["A"]
        S_id = skid["S_id"]
        W = skid["W"]
        while True:
            y = sample_vector(self.m, self.gamma2)
            w = self.mod_q(np.dot(A, y))
            c = self.H2(identity, w, message, self.k)
            z = np.dot(S_id, c) + y
            if np.all(np.abs(z) < (self.gamma2 - self.beta2)):
                break
        signature = {"z": z, "c": c, "W": W}
        return signature

    def Verify(self, mpk, signature, message, identity):
        A = mpk["A"]
        T = mpk["T"]
        z = signature["z"]
        c = signature["c"]
        W = signature["W"]
        C = self.H1(W, identity, self.k, self.kappa)
        Az = np.dot(A, z)
        TC = np.dot(T, C)
        # Note: W and T·C are matrices; here we assume the operations conform dimensionally.
        # For simplicity, treat (W + T·C)·c as a matrix-vector product.
        WC = W + TC
        WCc = np.dot(WC, c)
        w_prime = self.mod_q(Az - WCc)
        c_prime = self.H2(identity, w_prime, message, self.k)
        if np.all(np.abs(z) < (self.gamma2 - self.beta2)) and np.array_equal(c, c_prime):
            return True
        else:
            return False
    # --------------------------- END LB-IBS ALGORITHMS ---------------------------

    # --------------------------- HASH FUNCTIONS ---------------------------
    # Could be static, added to the class for functionality separation and clarity

    # H1: {0,1}* → {W in {-1,0,1}^(k×k) with ||W||₁ = κ}
    def H1(self, W, identity, k, kappa):
        # Use SHA-256 and then deterministically choose κ positions.
        data = W.tobytes() + identity.encode('utf-8')
        digest = hashlib.shake_256(data).digest(2*kappa)
        num_positions = k * k
        indices = []
        i = 0
        # Choose kappa distinct positions
        while len(indices) < kappa:
            start = (i * 4) % len(digest)
            idx = int.from_bytes(digest[start:start + 4], 'big') % num_positions
            if idx not in indices:
                indices.append(idx)
            i += 1
            if i * 4 > len(digest):
                digest = hashlib.shake_256(digest).digest(4*kappa)
                i = 0
        C = np.zeros((k, k), dtype=int)
        for j, idx in enumerate(indices):
            byte = digest[j]
            sign = 1 if (byte % 2 == 0) else -1
            row = idx // k
            col = idx % k
            C[row, col] = sign
        return C

    # H2: {0,1}* → {w in {-1,0,1}^(k) with ||w||₁ = κ}
    def H2(self, id_str, w, message, k):
        data = id_str.encode('utf-8') + w.tobytes() + message.encode('utf-8')
        digest = hashlib.shake_256(data).digest(4*k)
        indices = []
        for i in range(k):
            start = i * 4
            idx = int.from_bytes(digest[start:start + 4], 'big') % k
            if idx not in indices:
                indices.append(idx)
            else:
                j = i + 1
                while len(indices) < k:
                    start = j * 4
                    if start + 4 > len(digest):
                        digest += hashlib.shake_256(digest).digest(4*k)
                    new_idx = int.from_bytes(digest[start:start + 4], 'big') % k
                    if new_idx not in indices:
                        indices.append(new_idx)
                    j += 1
                break

        v = np.zeros((k, 1), dtype=int)
        for j, idx in enumerate(indices):
            byte = digest[j]  # use the first byte of each 4-byte chunk
            sign = 1 if (byte % 2 == 0) else -1
            v[idx, 0] = sign
        return v

    # --------------------------- END HASH FUNCTIONS ---------------------------