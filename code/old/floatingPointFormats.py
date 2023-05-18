import torch as pt 


'''

REAL NUMBERS

float32: 
    8 bit exponent, 23 bit mantissa, 1 bit sign

float16 / half precision: 
    5 bit exponent, 10 bit mantissa, 1 bit sign

float64 / double precision: 
    11 bit exponent, 53 bit mantissa



COMPLEX NUMBERS (at least in pytorch, presumably general)

complex64:
    Two real float32 numbers, one for real part - one for imag part.

complex32 / ComplexHalf:
    two float16 / half precision numbers - one for real part, one for imag  part

complex128 / ComplexDouble:
    two float64 / double precision numbers - one for real part, one for imag part


'''



max_exp32 = int('1111111',2)
max_mant32= int('1'*24,2) / (2**23)
max_exp16 = int('1111',2)
max_mant16 = int('1'*11,2)/(2**10)
max_exp64 = int(10*'1',2)
max_mant64 = int('1'*54,2)/(2**53)

print(f"Max float32 = {max_mant32*2**max_exp32}")
print(f"Max float16 = {max_mant16*2**max_exp16}")
#print(f"Max float64 = {max_mant64*2**max_exp64}")


x = pt.zeros(1,dtype=pt.complex128)
x[0]=max_mant32 * 2**max_exp32
print(x[0]+pt.tensor([1e37 ]))

print(x[0].type())

