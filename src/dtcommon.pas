unit DTCommon;

{$mode delphi}

interface

uses
  Classes, SysUtils, wincrt, Math, csvdocument;

const
  ERR_MSG_DIMENSION_MISMATCH = 'Dimension mismatch.';

type
  TFloatVector = array of real;
  TFloatMatrix = array of array of real;
  TIntVector = array of integer;
  TIntMatrix = array of array of integer;

  TCallbackFloat = function(x: real): real;
  TCallbackInt = function(x: real): Integer;
  TCallbackFloat1param = function(x: real; param1: real): real;
  TCallbackFloatArray = function(v: TFloatVector): TFloatVector;
  TCallbackString = function: string;



function Shape(mat: TFloatMatrix): TIntVector;
function CreateVector(size: integer; x: real): TFloatVector; overload;
function CreateVector(size: integer): TFloatVector; overload;
function CreateMatrix(row, col: integer; x: real): TFloatMatrix; overload;
function CreateMatrix(row, col: integer): TFloatMatrix; overload;
function MatToStr(mat: TFloatMatrix): string;
function VecToMat(v: TFloatVector): TFloatMatrix;
function ones(row, col: integer): TFloatMatrix;
function Transpose(mat: TFloatMatrix): TFloatMatrix;
function GetColumnVector(mat: TFloatMatrix; idx: integer): TFloatVector;
function GetColumn(mat: TFloatMatrix; idx: integer): TFloatMatrix;
function GetRange(mat: TFloatMatrix;
  rowFrom, colFrom, Height, Width: integer): TFloatMatrix;
procedure InsertRowAt(var A: TFloatMatrix; const Index: integer;
  const Value: TFloatVector);
procedure InsertColumnAt(var mat: TFloatMatrix; const index: integer;
  const Value: TFloatVector);


function FloatMatrixFromCSV(s: string): TFloatMatrix;


//function AddColumnAt(var mat:TFloatMatrix)

{
  Math function wrapper, to make them in compliant with the helper functions.
}
function Exp(x: real): real;
function Log(x: real): real;
function Pow(base, exponent: real): real;
function Round(x: real): Integer;

function ElementWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix; overload;
function ElementWise(func: TCallbackFloat; vec: TFloatVector): TFloatVector; overload;
function ElementWise(func: TCallbackInt; vec: TFloatVector): TIntVector;overload;

function ElementWise(func: TCallbackFloat1param; param1: real;
  mat: TFloatMatrix): TFloatMatrix; overload;

function RowWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix;
procedure AppendVector(var v: TFloatVector; x: real); overload;
procedure AppendVector(var v: TIntVector; x: integer); overload;
procedure PrintMatrix(Mat: TFloatMatrix);
procedure PrintVector(vec: TFloatVector); overload;
procedure PrintVector(vec: TIntVector); overload;


implementation

function stripNonAscii(const s: string): string;
var
  i, Count: integer;
begin
  SetLength(Result, Length(s));
  Count := 0;
  for i := 1 to Length(s) do
  begin
    if ((s[i] >= #32) and (s[i] <= #127)) or (s[i] in [#10, #13]) then
    begin
      Inc(Count);
      Result[Count] := s[i];
    end;
  end;
  SetLength(Result, Count);
end;

function FloatMatrixFromCSV(s: string): TFloatMatrix;
var
  i, j, m, n: integer;
  csvReader: TCSVDocument;
  res: TFloatMatrix;
begin
  csvReader := TCSVDocument.Create();
  csvReader.LoadFromFile(s);
  m := csvReader.RowCount;
  n := csvReader.ColCount[0];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
    begin
      res[i][j] := StrToFloat(stripNonAscii(csvReader.Cells[j, i]));
    end;
  end;
  Result := res;
end;

// Outputs the shape of a matrix, as [row, col]
function Shape(mat: TFloatMatrix): TIntVector;
var
  m, n: integer;
  res: TIntVector;
begin
  SetLength(res, 2);
  m := length(mat);
  if m = 0 then
    n := 0
  else
    n := length(mat[0]);
  res[0] := m;
  res[1] := n;
  Result := res;
end;

// create 'size'-length vector, and set all values to x
function CreateVector(size: integer; x: real): TFloatVector;
var
  i: integer;
  res: TFloatVector;
begin
  SetLength(res, size);
  for i := 0 to size - 1 do
    res[i] := x;
  Result := res;
end;

// create 'size'-length vector, and set all values randomly
function CreateVector(size: integer): TFloatVector;
var
  i: integer;
  res: TFloatVector;
begin
  SetLength(res, size);
  for i := 0 to size - 1 do
    res[i] := Random;
  Result := res;
end;

// create 'row' by 'col' matrix, and set all values to x
function CreateMatrix(row, col: integer; x: real): TFloatMatrix;
var
  i, j: integer;
  res: TFloatMatrix;
begin
  SetLength(res, row);
  for i := 0 to row - 1 do
  begin
    SetLength(res[i], col);
    for j := 0 to col - 1 do
      res[i][j] := x;
  end;
  Result := res;
end;

// create 'row' by 'col' matrix with random values
function CreateMatrix(row, col: integer): TFloatMatrix;
var
  i, j: integer;
  res: TFloatMatrix;
begin
  SetLength(res, row);
  for i := 0 to row - 1 do
  begin
    SetLength(res[i], col);
    for j := 0 to col - 1 do
      res[i][j] := Random;
  end;
  Result := res;
end;


// create 'row' by 'col' matrix with values of 1
function ones(row, col: integer): TFloatMatrix;
begin
  Result := CreateMatrix(row, col, 1);
end;

// create 1 by n matrix from vector
function VecToMat(v: TFloatVector): TFloatMatrix;
var
  res: TFloatMatrix;
begin
  SetLength(res, 1);
  res[0] := v;
  Result := res;
end;

// convert matrix to string
function MatToStr(mat: TFloatMatrix): string;
begin

end;

// Matrix transpose
function Transpose(mat: TFloatMatrix): TFloatMatrix;
var
  res: TFloatMatrix;
  mshape: TIntVector;
  m, n, i, j: integer;
begin
  mshape := shape(mat);
  m := mshape[0];
  n := mshape[1];
  SetLength(res, n);
  for i := 0 to n - 1 do
  begin
    SetLength(res[i], m);
    for j := 0 to m - 1 do

      res[i][j] := mat[j][i];
  end;
  Result := res;
end;


function GetColumnVector(mat: TFloatMatrix; idx: integer): TFloatVector;
var
  i, m: integer;
  res: TFloatVector;
begin
  m := Shape(mat)[0];
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := mat[i][idx];
  Result := res;
end;

function GetColumn(mat: TFloatMatrix; idx: integer): TFloatMatrix;
var
  i, m: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], 1);
    res[i][0] := mat[i][idx];
  end;
  Result := res;
end;

function GetRange(mat: TFloatMatrix;
  rowFrom, colFrom, Height, Width: integer): TFloatMatrix;
var
  res: TFloatMatrix;
  i, j: integer;
begin
  SetLength(res, Height);
  for i := rowFrom to Height - 1 do
  begin
    SetLength(res[i], Width);
    for j := colFrom to Width - 1 do
      res[i - rowFrom][j - colFrom] := mat[i][j];
  end;
  Result := res;
end;

procedure InsertRowAt(var A: TFloatMatrix; const Index: integer;
  const Value: TFloatVector);
var
  ALength: cardinal;
  TailElements: cardinal;
begin
  ALength := Length(A);
  Assert(Index <= ALength);
  SetLength(A, ALength + 1);
  TailElements := ALength - Index;
  if TailElements > 0 then
  begin
    Move(A[Index], A[Index + 1], SizeOf(real) * TailElements);
    Initialize(A[Index]);
    A[Index] := Value;
  end;
end;

procedure InsertColumnAt(var mat: TFloatMatrix; const index: integer;
  const Value: TFloatVector);
var
  tmp: TFloatMatrix;
begin
  tmp := Transpose(mat);
  InsertRowAt(tmp, index, Value);
  mat := Transpose(tmp);
end;




{
  Math function wrapper, to make them in compliant with the helper functions.
}
function Exp(x: real): real;
begin
  Result := system.exp(x);
end;

function Log(x: real): real;
begin
  Result := System.ln(x);
end;

function Round(x: real): Integer;
begin
  Result := System.round(x);
end;

function Pow(base, exponent: real): real;
begin
  Result := Math.power(base, exponent);
end;




{
  Helper functions to perform mapping
}
// apply element-wise operation on 2D array
function ElementWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix;
var
  i, j, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
      res[i][j] := func(mat[i][j]);
  end;
  Result := res;
end;

// apply element-wise operation on 2D array taking a real parameter
function ElementWise(func: TCallbackFloat1param; param1: real;
  mat: TFloatMatrix): TFloatMatrix;
var
  i, j, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
      res[i][j] := func(mat[i][j], param1);
  end;
  Result := res;
end;

// apply element-wise operation on array
function ElementWise(func: TCallbackFloat; vec: TFloatVector): TFloatVector;
var
  i, m: integer;
  res: TFloatVector;
begin
  m := Length(vec);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := func(vec[i]);
  Result := res;
end;

function ElementWise(func: TCallbackInt; vec: TFloatVector): TIntVector;
var
  i, m: integer;
  res: TIntVector;
begin
  m := Length(vec);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := func(vec[i]);
  Result := res;
end;

// apply row-wise operation on 2D array
function RowWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix;
var
  i, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    res[i] := ElementWise(func, mat[i]);
  end;
  Result := res;
end;

procedure AppendVector(var v: TFloatVector; x: real);
begin
  SetLength(v, Length(v) + 1);
  v[Length(v) - 1] := x;
end;

procedure AppendVector(var v: TIntVector; x: integer);
begin
  SetLength(v, Length(v) + 1);
  v[Length(v) - 1] := x;
end;


procedure PrintMatrix(Mat: TFloatMatrix);
var
  i, j, m, n: integer;
  s: string;
begin
  m := Shape(Mat)[0];
  n := Shape(Mat)[1];
  for i := 0 to m - 1 do
  begin
    s := '';
    for j := 0 to n - 1 do
    begin
      s := s + FloatToStr(Mat[i][j]);
      if j < n - 1 then
        s := s + ',    ';
    end;
    Writeln(s);
  end;
end;

procedure PrintVector(vec: TFloatVector);
var
  i, m: integer;
  s: string;
begin
  m := Length(vec);
  s := '';
  for i := 0 to m - 1 do
  begin
    s := s + FloatToStr(vec[i]);
    if i < m - 1 then
      s := s + ',    ';
  end;
  Writeln(s);
end;

procedure PrintVector(vec: TIntVector);
var
  i, m: integer;
  s: string;
begin
  m := Length(vec);
  s := '';
  for i := 0 to m - 1 do
  begin
    s := s + IntToStr(vec[i]);
    if i < m - 1 then
      s := s + ',    ';
  end;
  Writeln(s);
end;




end.
