# Darkteal

<p align="center">
  <img src="logo.png" alt="logo" width="100"/>
<p>

Darkteal is a library to perform some basic scientific computing in (object) pascal. **Still under a heavy development**. It is built purely in pascal language, with OpenBLAS backend for heavy matrix operations.

In the near future, darkteal is aimed to be a foundation to make the development of machine learning algorithm easier. That is why you will notice some neural network-related functions, such as data preprocessing function, a collection of activations, and loss functions.

**Important note:** This library is:
- In a very early development, thus, many missing functionality
- Not for production purpose, since it is not built for performance-critical applications

## Quick Start

### Usage
- Include darkteal's "src" folder into your project search path.
- Put libopenblas shared library (libopenblas.dll) in your project folder. Refer to OpenBLAS [installation guide](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide).

### Initializing matrices
Darkteal revolves around the usage of TDTMatrix. The TDTMatrix is essentially a wrapper of pascal's 2D dynamic array with OpenBLAS as the backend for some matrix operations. Darkteal provides helper functions to initialize matrices in various ways:
```pascal
program example;

uses
  wincrt,
  SysUtils,
  DTCore; // core include

var
  A, B, C: TDTMatrix;

begin
  DarkTealInit;                // Initialize darkteal (once)

  A := CreateMatrix(3, 4);     // 3x4 matrix with random values
  B := CreateMatrix(3, 4, 10); // 3x4 matrix filled with 10

  WriteLn('Matrix A: ');
  PrintMatrix(A);
  WriteLn();

  WriteLn('Matrix B: ');
  PrintMatrix(B);
end;
```

It is also possible to create a matrix by loading from a CSV file by using ```TDTMatrixFromCSV``` function. Keep in mind that this CSV loader is not a general purpose CSV loader, i.e., it is designed to load numerical data **only** for the sake of loading speed, at least for now. All numerical values will be converted into floating-point numbers.
```pascalWriteLn(Xtrain.Height);
  ...
  C := TDTMatrixFromCSV('yourfilename.csv');
  PrintMatrix(C);
  ...
```

Followings are the example of matrix operations:
```pascal
  // Transpose
  WriteLn('A transpose:');
  C := A.T;
  PrintMatrix(C);
  writeln();

  // Addition
  WriteLn('A + B:');
  C := Add(A, B); // or C := A + B;
  PrintMatrix(C);
  writeln();

  // Dot product
  WriteLn('<A, B>:');
  C := A.Dot(B.T);
  PrintMatrix(C);
  writeln();

  // Hadamard (element-wise) product
  WriteLn('A .* B:');
  C := Multiply(A, B); // or C := A * B
  PrintMatrix(C);

  ReadLn();  
```
Check out the documentation [here](https://ariaghora.github.io/darkteal/docs/)

### Some known issues
- **Some operations are painfully slow:** darkteal is still in a very early development. What you can do for now is making optimization on the compiler side, e.g., using "-O3" if you are using freepascal compiler.
- **Successful matrix operation despite of being under dimension mismatch:** Set the compiler to check assertion.

## License
This project is licensed under the MIT License
