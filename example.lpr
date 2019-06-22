program example;

uses
  {$IFDEF MSWINDOWS}
  wincrt,
  {$ENDIF}
  SysUtils,
  DTCore; // core include

var
  A, B, C: TDTMatrix;

begin
  DarkTealInit;

  A := CreateMatrix(3, 4);     // 3x4 matrix with random values
  B := CreateMatrix(3, 4, 10); // 3x4 matrix filled with 10

  WriteLn('Matrix A: ');
  PrintMatrix(A);
  WriteLn();

  WriteLn('Matrix B: ');
  PrintMatrix(B);
  WriteLn();

  // Transpose
  WriteLn('A transpose:');
  C := A.T;
  PrintMatrix(C);
  writeln();

  // Addition
  WriteLn('A + B:');
  C := A + B;
  PrintMatrix(C);
  writeln();

  // Dot product
  WriteLn('<A, B>:');
  C := A.Dot(B.T);
  PrintMatrix(C);
  writeln();

  // Hadamard (element-wise) product
  WriteLn('A .* B:');
  C := Multiply(A, B);
  PrintMatrix(C);
  writeln();

  ReadLn();
end.
