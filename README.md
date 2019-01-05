# Darkteal

Darkteal is a library to perform some basic scientific computing in (object) pascal. It is, as well, built **purely** in pascal language, with standard primitives. Darkteal provides some linear algebra functionality, such as vector and matrix operations including transpose, multiplication, dot products, etc.

Darkteal is aimed to be a foundation to make the development of machine learning algorithm easier. That is why you will notice some neural network-related functions, such as data preprocessing function, a collection of activations, and loss functions.

**Important note:** This library is:
- In a very early development, thus, many missing functionality
- Not for production, since it is not built for performance-critical applications

## Quick Start

### Installing
Include "dtcommon.pas", "dtlinalg.pas", "dtmlutils.pas", and "dtpreprocessing.pas" into your project, and you are ready to go.

### Try the example
```pascal
program example;

uses
  wincrt,
  SysUtils,
  DTCommon,
  DTLinAlg;

var
  A, B, C: TFloatMatrix;

begin
  A := CreateMatrix(3, 4);     // 3x4 matrix with random values
  B := CreateMatrix(3, 4, 10); // 3x4 matrix filled with 10

  WriteLn('Matrix A: ');
  PrintMatrix(A);
  WriteLn();

  WriteLn('Matrix B: ');
  PrintMatrix(B);
  WriteLn();

  // Addition
  WriteLn('A + B:');
  C := Add(A, Transpose(B));
  PrintMatrix(C);
  writeln();  

  // Dot product
  WriteLn('<A, B>:');
  C := DotProduct(A, Transpose(B));
  PrintMatrix(C);
  writeln();

  // Hadamard (element-wise) product
  WriteLn('A .* B:');
  C := Multiply(A, B);
  PrintMatrix(C);
  writeln();

  ReadLn();  
end;
```

### Some known issues
- **Some operations are painfully slow:** darkteal is still in a very early development. What you can do for now is making optimization on the compiler side, e.g., using "-O3" if you are using freepascal compiler.

## License
This project is licensed under the MIT License

## Acknowledgements