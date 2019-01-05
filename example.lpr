program example;

uses
  wincrt,
  SysUtils,
  DTCommon,
  DTLinAlg,
  DTPreprocessing;

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
end.
