unit DTCommon;

{$mode delphi}
//{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, wincrt, Math, dynlibs, csvdocument;

const
  ERR_MSG_DIMENSION_MISMATCH = 'Dimension mismatch.';

type
  TFloatVector = array of double;
  TFloatMatrix = array of array of double;
  TIntVector = array of integer;
  TIntMatrix = array of array of integer;
  TCallbackFloat = function(x: double): double;
  TCallbackInt = function(x: double): integer;
  TCallbackFloat1param = function(x: double; param1: double): double;
  TCallbackFloatArray = function(v: TFloatVector): TFloatVector;
  TCallbackString = function: string;

  CBLAS_ORDER = (CblasRowMajor = 101, CblasColMajor = 102);
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113);

  _sdot = function(N: longint; X: PSingle; incX: longint; Y: PSingle;
    incY: longint): single; cdecl;
  _ddot = function(N: longint; X: TFloatVector; incX: longint;
    Y: TFloatVector; incY: longint): double; cdecl;
  _dgemm = procedure(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
    TransB: CBLAS_TRANSPOSE; M: longint; N: longint; K: longint;
    alpha: double; A: TFloatVector; lda: longint; B: TFloatVector;
    ldb: longint; beta: double; C: TFloatVector; ldc: longint); cdecl;
  _sgemm = procedure(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
    TransB: CBLAS_TRANSPOSE; M: longint; N: longint; K: longint;
    alpha: single; A: Psingle; lda: longint; B: Psingle; ldb: longint;
    beta: single; C: Psingle; ldc: longint); cdecl;

  TDTMatrix = record
    val: TFloatMatrix;
    class operator Implicit(mat: TFloatMatrix): TDTMatrix;
    class operator Explicit(mat: TFloatMatrix): TDTMatrix;
    class operator Add(A, B: TDTMatrix): TDTMatrix;
    class operator Subtract(A, B: TDTMatrix): TDTMatrix;
    class operator Multiply(x: double; A: TDTMatrix): TDTMatrix; overload;
    class operator Multiply(A, B: TDTMatrix): TDTMatrix; overload;
    class operator Divide(A, B: TDTMatrix): TDTMatrix; overload;
    class operator Divide(A: TDTMatrix; x: double): TDTMatrix; overload;
    function T: TDTMatrix;
    function Shape: TIntVector;
    function ToStr: string;
    function Dot(A: TDTMatrix): TDTMatrix;
    function Sum: double; overload;
    function Sum(dims: integer): TDTMatrix; overload;
    function GetRange(rowDrom, colFrom, Height, Width: integer): TDTMatrix;
    function Flatten: TDTMatrix;
  end;

function Shape(mat: TFloatMatrix): TIntVector;
function CreateVector(size: integer; x: double): TFloatVector; overload;
function CreateVector(size: integer): TFloatVector; overload;
function CreateMatrix(row, col: integer; x: double): TFloatMatrix; overload;
function CreateMatrix(row, col: integer): TFloatMatrix; overload;
function MatToStr(mat: TFloatMatrix): string;
function VecToMat(v: TFloatVector): TFloatMatrix;
function ones(row, col: integer): TFloatMatrix;
function Transpose(mat: TFloatMatrix): TFloatMatrix;
function GetColumnVector(mat: TFloatMatrix; idx: integer): TFloatVector;
function GetColumn(mat: TFloatMatrix; idx: integer): TFloatMatrix;
function GetRange(mat: TFloatMatrix;
  rowFrom, colFrom, Height, Width: integer): TFloatMatrix;
function Flatten(mat: TFloatMatrix): TFloatMatrix;
procedure InsertRowAt(var A: TFloatMatrix; const Index: integer;
  const Value: TFloatVector);
procedure InsertColumnAt(var mat: TFloatMatrix; const index: integer;
  const Value: TFloatVector);

function FloatMatrixFromCSV(s: string): TFloatMatrix;
function TDTMatrixFromCSV(f: string): TDTMatrix;

function Exp(x: double): double;
function Log(x: double): double;
function Pow(base, exponent: double): double;
function Round(x: double): integer;

function ElementWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix; overload;
function ElementWise(func: TCallbackFloat; vec: TFloatVector): TFloatVector; overload;
function ElementWise(func: TCallbackInt; vec: TFloatVector): TIntVector; overload;
function ElementWise(func: TCallbackFloat1param; param1: double;
  mat: TFloatMatrix): TFloatMatrix; overload;

function RowWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix;
procedure AppendVector(var v: TFloatVector; x: double); overload;
procedure AppendVector(var v: TIntVector; x: integer); overload;
procedure PrintMatrix(Mat: TFloatMatrix);
procedure PrintVector(vec: TFloatVector); overload;
procedure PrintVector(vec: TIntVector); overload;

var

  blas_sdot: _sdot;
  blas_ddot: _ddot;
  blas_dgemm: _dgemm;
  blas_sgemm: _sgemm;
  libHandle: TLibHandle;


implementation

uses DTLinAlg;

//========= End of TDMatrix implementations =========//

class operator TDTMatrix.Explicit(mat: TFloatMatrix): TDTMatrix;
begin
  Result.val := mat;
end;

class operator TDTMatrix.Implicit(mat: TFloatMatrix): TDTMatrix;
begin
  Result.val := mat;
end;

class operator TDTMatrix.add(A, B: TDTMatrix): TDTMatrix;
begin
  Result := Add(A.val, B.val);
end;

class operator TDTMatrix.Subtract(A, B: TDTMatrix): TDTMatrix;
begin
  Result := Subtract(A.val, B.val);
end;

class operator TDTMatrix.Multiply(x: double; A: TDTMatrix): TDTMatrix;
begin
  Result := Multiply(x, A.val);
end;

class operator TDTMatrix.Multiply(A, B: TDTMatrix): TDTMatrix;
begin
  Result := Multiply(A.val, B.val);
end;

class operator TDTMatrix.Divide(A, B: TDTMatrix): TDTMatrix;
begin
  Result := DTLinAlg.Divide(A.val, B.val);
end;

class operator TDTMatrix.Divide(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := DTLinAlg.Divide(A.val, x);
end;

function TDTMatrix.T: TDTMatrix;
begin
  Result.val := Transpose(self.val);
end;

function TDTMatrix.Shape: TIntVector;
begin
  Result := DTCommon.Shape(Self.val);
end;

function TDTMatrix.ToStr: string;
begin
  Result := MatToStr(self.val);
end;

function TDTMatrix.Dot(A: TDTMatrix): TDTMatrix;
begin
  Result.val := DotProduct(Self.val, A.val);
end;

function TDTMatrix.Sum: double;
begin
  Result := DTLinAlg.Sum(self.val);
end;

function TDTMatrix.Sum(dims: integer): TDTMatrix;
begin
  Result.val := DTLinAlg.Sum(self.val, dims);
end;

function TDTMatrix.GetRange(rowDrom, colFrom, Height, Width: integer): TDTMatrix;
begin
  Result := DTCommon.GetRange(Self.val, rowDrom, colFrom, Height, Width);
end;

function TDTMatrix.Flatten: TDTMatrix;
begin
  Result := DTCommon.Flatten(self.val);
end;

//========= End of TDMatrix implementations =========//

procedure DarkTealInit;
begin
  libHandle := LoadLibrary('libopenblas.dll');
  Pointer(@blas_sdot) := GetProcedureAddress(libHandle, 'cblas_sdot');
  Pointer(@blas_ddot) := GetProcedureAddress(libHandle, 'cblas_ddot');
  Pointer(@blas_dgemm) := GetProcedureAddress(libHandle, 'cblas_dgemm');
  Pointer(@blas_sgemm) := GetProcedureAddress(libHandle, 'cblas_sgemm');
end;

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

function TDTMatrixFromCSV(f: string): TDTMatrix;
var
  tfIn: TextFile;
  s, row: string;
  i, cntRow, cntCol: integer;
  res: TDTMatrix;
begin
  assignfile(tfIn, f);
  try
    Reset(tfIn); // open file
    cntRow := 0;
    s := '';
    while not EOF(tfIn) do
    begin
      Readln(tfIn, row);
      Inc(cntRow);
      SetLength(res.val, cntRow);
      cntCol := 0;
      for i := 1 to Length(row) do
      begin
        if (row[i] <> ',') then
          s := s + row[i]
        else
        begin
          Inc(cntCol);
          SetLength(res.val[cntRow - 1], cntCol);
          res.val[cntRow - 1][cntCol - 1] := StrToFloat(s);
          s := '';
        end;
        if i = Length(row) then
        begin
          Inc(cntCol);
          SetLength(res.val[cntRow - 1], cntCol);
          res.val[cntRow - 1][cntCol - 1] := StrToFloat(s);
        end;
      end;
      s := '';
    end;
    CloseFile(tfIn); // close file
  except
    on E: EInOutError do
      writeln('File handling error occurred. Details: ', E.Message);
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
function CreateVector(size: integer; x: double): TFloatVector;
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
function CreateMatrix(row, col: integer; x: double): TFloatMatrix;
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
var
  i, j: integer;
  matShape: TIntVector;
begin
  Result := '[';
  matShape := Shape(mat);
  for i := 0 to matShape[0] - 1 do
  begin
    //if i = 0 then
    Result := Result + '[';
    for j := 0 to matShape[1] - 1 do
    begin
      Result := Result + FloatToStr(mat[i][j]);
      if j < matShape[1] - 1 then
        Result := Result + ', '
      else
        Result := Result + ']';
    end;
    if i < matShape[0] - 1 then
      Result := Result + ',' + sLineBreak;
  end;
  Result := Result + ']';
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
  for i := rowFrom to rowFrom + Height - 1 do
  begin
    SetLength(res[i], Width);
    for j := colFrom to colFrom + Width - 1 do
      res[i - rowFrom][j - colFrom] := mat[i][j];
  end;
  Result := res;
end;

function Flatten(mat: TFloatMatrix): TFloatMatrix;
var
  i, j, idx, m, n: integer;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  SetLength(Result, 1);
  SetLength(Result[0], m * n);
  idx := 0;
  for i := 0 to m - 1 do
    for j := 0 to n - 1 do
    begin
      Result[0][idx] := mat[i][j];
      Inc(idx);
    end;
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
    Move(A[Index], A[Index + 1], SizeOf(double) * TailElements);
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
function Exp(x: double): double;
begin
  Result := system.exp(x);
end;

function Log(x: double): double;
begin
  Result := System.ln(x);
end;

function Round(x: double): integer;
begin
  Result := System.round(x);
end;

function Pow(base, exponent: double): double;
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

// apply element-wise operation on 2D array taking a Double parameter
function ElementWise(func: TCallbackFloat1param; param1: double;
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

procedure AppendVector(var v: TFloatVector; x: double);
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

//showmess




end.
