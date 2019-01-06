unit DTLinAlg;

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCommon;

function Add(mat1, mat2: TFloatMatrix): TFloatMatrix; overload;
function Add(v1, v2: TFloatVector): TFloatVector; overload;
function Subtract(mat1, mat2: TFloatMatrix): TFloatMatrix; overload;
function Subtract(v1, v2: TFloatVector): TFloatVector; overload;
function DotProduct(v1: TFloatVector; v2: TFloatVector): real; overload;
function DotProduct(m1: TFloatMatrix; m2: TFloatMatrix): TFloatMatrix; overload;
// same kinds
function Multiply(v1, v2: TFloatVector): TFloatVector; overload;
function Multiply(m1, m2: TFloatMatrix): TFloatMatrix; overload;
function Multiply(x: real; mat: TFloatMatrix): TFloatMatrix; overload;
// vector-matrix --> broadcasting
function Multiply(v: TFloatVector; mat: TFloatMatrix): TFloatMatrix; overload;

function Divide(v1, v2: TFloatVector): TFloatVector; overload;
function Divide(v: TFloatVector; x: real): TFloatVector; overload;
function Divide(x: real; v: TFloatVector): TFloatVector; overload;
function Divide(m1, m2: TFloatMatrix): TFloatMatrix; overload;
function Sum(v: TFloatVector): real; overload;
function Sum(mat: TFloatMatrix): real; overload;
function Sum(mat: TFloatMatrix; dims: integer): TFloatMatrix; overload;

implementation


{
  A collection of functions to perform varoius linear algebra operations.
}

// Vector-Vector addition
function Add(v1, v2: TFloatVector): TFloatVector;
var
  i, m: integer;
  res: TFloatVector;
begin
  m := Length(v1);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := v1[i] + v2[i];
  Result := res;
end;

// Matrix-matrix addition
function Add(mat1, mat2: TFloatMatrix): TFloatMatrix;
var
  i, j, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat1)[0];
  n := Shape(mat1)[1];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
      res[i][j] := mat1[i][j] + mat2[i][j];
  end;
  Result := res;
end;



// Matrix-matrix subtraction
function Subtract(mat1, mat2: TFloatMatrix): TFloatMatrix;
var
  i, j, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat1)[0];
  n := Shape(mat1)[1];
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
      res[i][j] := mat1[i][j] - mat2[i][j];
  end;
  Result := res;
end;

// vector-vector subtraction
function Subtract(v1, v2: TFloatVector): TFloatVector;
var
  i, m: integer;
  res: TFloatVector;
begin
  m := Length(v1);
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    res[i] := v1[i] - v2[i];
  end;
  Result := res;
end;

// Vector dot product
function DotProduct(v1: TFloatVector; v2: TFloatVector): real;
var
  i: integer;
  res: real = 0;
begin
  //Assert(Length(v1) = Length(v2), ERR_MSG_DIMENSION_MISMATCH);
  for i := 0 to Length(v1) - 1 do
    res := res + v1[i] * v2[i];
  Result := res;
end;

// Matrix dot product
function DotProduct(m1: TFloatMatrix; m2: TFloatMatrix): TFloatMatrix;
var
  i, j: integer;
  m, n: integer;
  res: TFloatMatrix;
  m2trans: TFloatMatrix;
begin
  m := Shape(m1)[0];
  n := Shape(m2)[1];
  m2trans := Transpose(m2);
  SetLength(res, m);
  for i := 0 to m - 1 do
  begin
    SetLength(res[i], n);
    for j := 0 to n - 1 do
      res[i][j] := DotProduct(m1[i], m2trans[j]);
  end;
  Result := res;
end;

// vector-vector hadamard product
function Multiply(v1, v2: TFloatVector): TFloatVector;
var
  i, m, n: integer;
  res: TFloatVector;
begin
  m := Length(v1);
  n := Length(v2);
  Assert(m = n, ERR_MSG_DIMENSION_MISMATCH);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := v1[i] * v2[i];
  Result := res;
end;

// vector-matrix multiplication
function Multiply(v: TFloatVector; mat: TFloatMatrix): TFloatMatrix;
var
  i, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  res := CreateMatrix(m, n, 0);
  for i := 0 to m - 1 do
    res[i] := Multiply(v, mat[i]);
  Result := res;
end;

// matrix-matrix hadamard product
function Multiply(m1, m2: TFloatMatrix): TFloatMatrix;
var
  i, m, n: integer;
  res: TFloatMatrix;
begin
  // if either one has row=1, then use the broadcast instead
  if (Length(m1) = 1) and (Length(m2) > 1) then
    Result := Multiply(m1[0], m2)
  else if (Length(m1) > 1) and (Length(m2) = 1) then
    Result := Multiply(m2[0], m1)
  else
  begin
    // otherwise, perform usual hadamard
    m := Length(m1);
    n := Length(m2);
    //Assert(m = n, ERR_MSG_DIMENSION_MISMATCH);
    SetLength(res, m);
    for i := 0 to m - 1 do
      res[i] := multiply(m1[i], m2[i]);
    Result := res;
  end;
end;

// scalar-matrix multiplication
function Multiply(x: real; mat: TFloatMatrix): TFloatMatrix;
var
  i, j, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  res := CreateMatrix(m, n);
  for i := 0 to m - 1 do
    for j := 0 to n - 1 do
      res[i][j] := x * mat[i][j];
  Result := res;
end;


// vector-vector division
function Divide(v1, v2: TFloatVector): TFloatVector;
var
  i, m, n: integer;
  res: TFloatVector;
begin
  m := Length(v1);
  n := Length(v2);
  Assert(m = n, ERR_MSG_DIMENSION_MISMATCH);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := v1[i] / v2[i];
  Result := res;
end;

// matrix-vector division
function Divide(mat: TFloatMatrix; v: TFloatVector): TFloatMatrix;
var
  i, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  res := CreateMatrix(m, n, 0);
  for i := 0 to m - 1 do
    res[i] := Divide(mat[i], v);
  Result := res;
end;

// matrix-matrix division
function Divide(m1, m2: TFloatMatrix): TFloatMatrix;
var
  i, m, n: integer;
  res: TFloatMatrix;
begin
  //if either one has row=1, then use the broadcast instead
  if (Length(m1) > 1) and (Length(m2) = 1) then
    Result := Divide(m1, m2[0])
  else
  begin
    // otherwise, perform usual hadamard
    m := Length(m1);
    n := Length(m2);
    //Assert(m = n, ERR_MSG_DIMENSION_MISMATCH);
    SetLength(res, m);
    for i := 0 to m - 1 do
    begin
      SetLength(res[i], n);
      res[i] := Divide(m1[i], m2[i]);
    end;
    Result := res;
  end;
end;

// vector-scalar division
function Divide(v: TFloatVector; x: real): TFloatVector;
var
  i, m: integer;
  res: TFloatVector;
begin
  m := Length(v);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := v[i] / x;
  Result := res;
end;

// scalar-vector division
function Divide(x: real; v: TFloatVector): TFloatVector;
var
  i, m: integer;
  res: TFloatVector;
begin
  m := Length(v);
  SetLength(res, m);
  for i := 0 to m - 1 do
    res[i] := x / v[i];
  Result := res;
end;


// sum of vector
function Sum(v: TFloatVector): real;
begin
  Result := Math.sum(v);
end;

// sum of matrix
function Sum(mat: TFloatMatrix): real;
var
  i: integer;
  res: real;
begin
  res := 0;
  for i := 0 to Length(mat) - 1 do
    res := res + Math.sum(mat[i]);
  Result := res;
end;

// sum of matrix by dimension
function Sum(mat: TFloatMatrix; dims: integer): TFloatMatrix;
var
  i, m, n: integer;
  res: TFloatVector;
  tmp: TFloatMatrix;
begin
  if dims = 0 then
  begin
    m := length(mat);
    n := shape(mat)[1];
    res := CreateVector(n, 0);
    for i := 0 to m - 1 do
    begin
      res := Add(res, mat[i]);
    end;
    Result := VecToMat(res);
  end
  else
  begin
    tmp := Transpose(mat);
    m := length(tmp);
    n := shape(tmp)[1];

    res := CreateVector(n, 0);
    for i := 0 to m - 1 do
    begin
      res := Add(res, tmp[i]);
    end;
    Result := Transpose(VecToMat(res));
  end;


  //res := 0;
  //for i := 0 to Length(mat) - 1 do
  //  res := res + Math.sum(mat[i]);
  //Result := res;
end;

end.

