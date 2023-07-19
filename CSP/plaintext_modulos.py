import numpy as np
from Pyfhel import Pyfhel
HE = Pyfhel()           # Creating empty Pyfhel object

# HE.contextGen(scheme='bgv', n=2**14, t_bits=20)  # Generate context for 'bfv'/'bgv'/'ckks' scheme

bgv_params = {
    'scheme': 'BGV',    # can also be 'bgv'
    'n': 2**13,         # Polynomial modulus degree, the num. of slots per plaintext,
                        #  of elements to be encoded in a single ciphertext in a
                        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
                        #  Typ. 2^D for D in [10, 16]
    't': 65537,         # Plaintext modulus. Encrypted operations happen modulo t
                        #  Must be prime such that t-1 be divisible by 2^N.       # Number of bits in t. Used to generate a suitable value
                        #  for t. Overrides t if specified.
    'sec': 128,         # Security parameter. The equivalent length of AES key in bits.
                        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
                        #  More means more security but also slower computation.
}
HE.contextGen(**bgv_params)  # Generate context for bgv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()       # Rotate key generation --> Allows rotation/shifting
HE.relinKeyGen()        # Relinearization key generation

print("\n2. Pyfhel FHE context generation")
print(f"\t{HE}")