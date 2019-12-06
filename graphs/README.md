## Data Format

#### *.tensors.dat
Each line is a tensor in **column major** format, e.g. for a 2x2 matrix, the data is
```
real_11 imag_11 real_21 imag_21 real_12 imag_12 real_22 imag_22
```

#### *.sizes.dat
Store the size information of above tensors. Each line is a tuple of size.

#### *.labels.dat
Each line is a labels (integer) for tensor, the information used for contraction.
