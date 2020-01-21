<div align="center">
<img src="assets/noe-txt.png" alt="logo" width="200px"></img>
</div>

This is my experimental project to understand the mechanism behind n-dimensional array: how is its layout in memory, how to do indexing, etc. Furthermore, I am also learning what happens under the hood of the automatic differentiation for a neural network in near future. That said, noe (In Korean: 뇌, means “brain”) is developed towards the implementation of neural networks (therefore the name). So, this project might be a good basis for it.

> This project used to be "darkteal". Please check the `master` branch for darkteal's code. There are some machine learning stuffs there. It is not abandoned. It is being upgraded with a better data representation... ;)

## Table of contents
- [Quickstart](#quickstart)
- [Declaring and initializing tensors](#Declaring-and-initializing-tensors)
- [Accessing tensor values](#Accessing-tensor-values)
- [Some basic math operations](#Some-basic-math-operations)
- [Einsum](#einsum)
  - [Issues on `einsum`](#issues-on-einsum)
- [Other considerations](#Other-considerations)

## Quickstart
### Basic setup
- Add `noe/src` into the include search path, and you are good to go with the basic.
  - In Lazarus IDE: Project > Project Options > Compiler Options > Paths. Then add the path to `noe/src` in "Other unit files (-Fu)".
### BLAS backend
Noe provides optional (**but recommended**) integration with basic linear algebra subroutine (BLAS) to accelerate several functions, such as matrix multiplication (`MatMul`). Noe uses BLAS implementation based on OpenBLAS. To install OpenBLAS:
- On Linux:
   - Debian/Ubuntu/Kali: run `apt install libopenblas-base`.
- On Windows:
   - Provide the libopenblas.dll. 
- On OSX:
   - Provide the libopenblas.dylib.

It is always recommended to compile the so/dll/dylib yourself, **especially for Windows**. Some precompiled one might be available out there, but I cannot guarantee the compatibility. Please refer to [this link](https://github.com/xianyi/OpenBLAS) for more complete documentation about compiling OpenBLAS.

## Declaring and initializing tensors
```delphi
uses
  noe.core, // --> main unit
  noe.math  // --> extending standard math unit

var
  A, B, C: TTensor; 
```

A tensor filled with a specific value:
```delphi
{ 2x3 tensor filled with 1 }
A := FullTensor([2, 3], 1);
PrintTensor(A);
```

```
[[1.00, 1.00, 1.00]
 [1.00, 1.00, 1.00]]
```
A tensor of random values:
```delphi
{ 2x2x3 tensor filled with randomized values }
B := FullTensor([2, 2, 3]);
PrintTensor(B);
```

```
[[[0.83, 0.03, 0.96]
  [0.68, 0.91, 0.04]]

 [[0.45, 0.83, 0.70]
  [0.56, 0.31, 0.77]]]
```
A tensor with specified values:
```delphi
{ 2x3 tensor filled with values specified }
C := FullTensor(
  [2, 3],      //--> target shape

  [1., 2., 3., //--> the data
   4., 5., 6.] //
);
PrintTensor(C);

WriteLn;

{ Reshape C into a 6x1 tensor }
C.Reshape([6, 1]);
PrintTensor(C);
```

```
[[1.00, 2.00, 3.00]
 [4.00, 5.00, 6.00]]

[[1.00]
 [2.00]
 [3.00]
 [4.00]
 [5.00]
 [6.00]]
```
A helper function `RangeF(n)` is provided to generate an ```array of float``` containing values from 0 to n-1. It can be paired with `FullTensor` to initialize a tensor:
```delphi
A := FullTensor([3,4,5], RangeF(60));
PrintTensor(A);
```
```
[[[ 0.00,  1.00,  2.00,  3.00,  4.00]
  [ 5.00,  6.00,  7.00,  8.00,  9.00]
  [10.00, 11.00, 12.00, 13.00, 14.00]
  [15.00, 16.00, 17.00, 18.00, 19.00]]

 [[20.00, 21.00, 22.00, 23.00, 24.00]
  [25.00, 26.00, 27.00, 28.00, 29.00]
  [30.00, 31.00, 32.00, 33.00, 34.00]
  [35.00, 36.00, 37.00, 38.00, 39.00]]

 [[40.00, 41.00, 42.00, 43.00, 44.00]
  [45.00, 46.00, 47.00, 48.00, 49.00]
  [50.00, 51.00, 52.00, 53.00, 54.00]
  [55.00, 56.00, 57.00, 58.00, 59.00]]]
```
## Accessing tensor values
To access the value of a tensor we can use multidimensional indexing:
```delphi
A := FullTensor([3, 2, 3]);
WriteLn('A:');
PrintTensor(A);

WriteLn(sLineBreak + 'A at index [2]:');
PrintTensor(A.GetAt([2]));

WriteLn(sLineBreak + 'A at index [0, 1]:');
PrintTensor(A.GetAt([0, 1]));

WriteLn(sLineBreak + 'A at index [1, 1, 0]:');
PrintTensor(A.GetAt([1, 1, 0]));            
```

```
A:
[[[0.97, 0.47, 0.57]
  [0.08, 0.51, 0.13]]

 [[0.43, 0.67, 0.55]
  [0.93, 0.86, 0.84]]

 [[0.00, 0.32, 0.03]
  [0.57, 0.95, 0.32]]]

A at index [2]:
[[0.00, 0.32, 0.03]
 [0.57, 0.95, 0.32]]

A at index [0, 1]:
[0.08, 0.51, 0.13]

A at index [1, 1, 0]:
0.93
```
## Some basic math operations
Several basic math operations on tensors are also supported.
```delphi
A := FullTensor([3, 3], 1);
B := FullTensor([3, 3]);
WriteLn('A:');
PrintTensor(A);
WriteLn('B:');
PrintTensor(B);

WriteLn('A + B:');
PrintTensor(A + B);

WriteLn('A - B:');
PrintTensor(A - B);

WriteLn('A * B:');
PrintTensor(A * B);
```
And some others:
```delphi
A := FullTensor([3,3]) + FullTensor([3, 3], 1);
PrintTensor( Log10(A) );
PrintTensor( Log2(A) );

A := FullTensor(
  [2, 2],

  [ 0., 30.,
   45., 90.]
);

A := DegToRad(A); // Also check RadToDeg(A)
PrintTensor( Sin(A) );
PrintTensor( Cos(A) );
PrintTensor( Tan(A) );

A := FullTensor(
  [2, 2],
  [1., 2.,
   3., 4.]
);
A := A ** 2;
PrintTensor(A); 
```
Please check `noe.math.pas` for more covered functionalities.

## Einsum!
I also implemented `Einsum` (Einstein's summation convention) function. It mirrors (subset of) numpy's einsum functionality. Using the `Einsum`, many common multi-dimensional, linear algebraic array operations can be represented in a simple fashion. 
```delphi
A := FullTensor(
  [3, 3],

  [1, 2, 3,
   4, 5, 6,
   7, 8, 9]
);
B := FullTensor([3, 4], 2);

WriteLn('A:'); printtensor(A); WriteLn();
WriteLn('B:'); printtensor(B); WriteLn();

WriteLn();

WriteLn('dot product AB:');
printtensor(Einsum('ij,jk->ik', [A, B]));
WriteLn();

WriteLn('element-wise product A o B:');
printtensor(Einsum('ij,ij->ij', [A, B]));
WriteLn();

WriteLn('diagonal of A:');
printtensor(Einsum('ii->i', [A]));
WriteLn();

WriteLn('sum of diagonal of A:');
printtensor(Einsum('ii', [A]));
WriteLn();

WriteLn('A transposed:');
printtensor(Einsum('ij->ji', [A]));
WriteLn();

WriteLn('sum of A along 1st dimension:');
printtensor(Einsum('ij->i', [A]));
WriteLn();

WriteLn('sum of A along 2nd dimension:');
printtensor(Einsum('ij->j', [A]));
WriteLn();
```
It also works on the operations of higher rank tensors:
```delphi
A := FullTensor(
  [2, 2, 3],

  [1, 2, 3,
   4, 5, 6,

   4, 5, 6,
   7, 8, 9]
);
B := FullTensor([3, 4], 2);

WriteLn('A:'); printtensor(A); WriteLn();
WriteLn('B:'); printtensor(B); WriteLn();

WriteLn('batch matrix multiplication of A & B:');
printtensor(Einsum('aij,jk->aik', [A, B]));
WriteLn();

WriteLn('A transposed w.r.t. dimension 2 & 3:');
printtensor(Einsum('ijk->ikj', [A]));
WriteLn();

{ Note: RangeF(60) generates an array of float from 0 to 59 }
A := FullTensor([3,4,5], RangeF(60)); 
B := FullTensor([4,3,2], RangeF(24));

WriteLn('tensor contraction:');
printTensor(Einsum('ijk,jil->kl', [A, B]));
WriteLn();
```
### Issues on `einsum`
- Slow. The current implementation is painfully slow. Do not use it too much.
- The `Einsum` implementation is yet to be ready. There are some known notations which will output undesirable result:
  - Sum of entries `Einsum('ij->', [A])` 
  - Bilinear transformation `Einsum('ik,jkl,il->ij', [A, B, C])` 

Please have a try, and open an issue if you find more nonfunctional notations. I will appreciate.


## Other considerations

- No complex number handling yet.
- Performance is not of my primary concern, at least for now. Kindly note that I am a firm believer of a saying “premature optimization is the root of all evil". I want simply a quick proof of concept of what I am learning. But it is not too shabby either. Don't worry.