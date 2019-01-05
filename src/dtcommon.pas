unit DTCommon;

{$mode delphi}

interface

uses
  Classes, SysUtils, wincrt, csvdocument;

const
  ERR_MSG_DIMENSION_MISMATCH = 'Dimension mismatch.';

type
  TFloatVector = array of real;
  TFloatMatrix = array of array of real;
  TIntVector = array of integer;
  TIntMatrix = array of array of integer;

  TCallbackFloat = function(x: real): real;
  TCallbackFloatArray = function(v: TFloatVector): TFloatVector;
  TCallbackString = function: string;


function FloatMatrixFromCSV(s: string): TFloatMatrix;
function Shape(mat: TFloatMatrix): TIntVector;
function CreateMatrix(row, col: integer; x: real): TFloatMatrix; overload;
function CreateMatrix(row, col: integer): TFloatMatrix; overload;
function ones(row, col: integer): TFloatMatrix;
function GetColumn(mat: TFloatMatrix; idx: integer): TFloatVector;

{
  Math function wrapper, to make them in compliant with the helper functions.
}
function Exp(x: real): real;
function Log(x: real): real;

function ElementWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix; overload;
function ElementWise(func: TCallbackFloat; vec: TFloatVector): TFloatVector; overload;
function RowWise(func: TCallbackFloat; mat: TFloatMatrix): TFloatMatrix;

procedure AppendVector(var v: TFloatVector; x: real); overload;
procedure AppendVector(var v: TIntVector; x: integer); overload;

procedure PrintMatrix(Mat: TFloatMatrix);
procedure PrintVector(vec: TFloatVector);


implementation

function FloatMatrixFromCSV(s: string): TFloatMatrix;
var
  i, j, m, n: integer;
  csvReader: TCSVDocument;
  res: TFloatMatrix;
begin
  csvReader := TCSVDocument.Create();
  csvReader.LoadFromFile('iris.csv');
  m := csvReader.RowCount;
  n := csvReader.ColCount[0];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
    begin
      res[i][j] := StrToFloat(csvReader.Cells[j, i]);
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

// create 'row' by 'col' matrix, and set all values to x
function CreateMatrix(row, col: integer; x: real): TFloatMatrix; overload;
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

function GetColumn(mat: TFloatMatrix; idx: integer): TFloatVector;
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




end.
