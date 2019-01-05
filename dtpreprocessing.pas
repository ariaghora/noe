unit DTPreprocessing;

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, URadixSort, DTCommon, DTLinAlg;

function MinMaxNormalize(mat: TFloatMatrix): TFloatMatrix;
function OneHotEncode(y: TIntVector): TFloatMatrix;

implementation

function MinMaxNormalize(mat: TFloatMatrix): TFloatMatrix;
var
  maxes, mins, range: TFloatVector;
  i, m, n: integer;
  res: TFloatMatrix;
begin
  m := Shape(mat)[0];
  n := Shape(mat)[1];
  SetLength(maxes, n);
  SetLength(mins, n);

  res := CreateMatrix(m, n);

  for i := 0 to n - 1 do
  begin
    maxes[i] := Math.MaxValue(GetColumn(mat, i));
    mins[i] := Math.MinValue(GetColumn(mat, i));
  end;
  range := Subtract(maxes, mins);

  for i := 0 to m - 1 do
    res[i] := Divide(Subtract(mat[i], mins), range);
  Result := res;
end;

{
  OneHotEncode encodes categorical integer features as a one-hot numeric array.
  It creates a binary column for each category and returns a sparse matrix.
  The input should be a TIntVector.
}
function OneHotEncode(y: TIntVector): TFloatMatrix;
var
  ySorted, unique: TIntVector;
  res: TFloatMatrix;
  i, j, c, currFound, cntUnique: integer;
begin
  {
    perform sorting on the (copied) 'y' to reduce the time complexity to
    O(log n) for finding unique class labels.
  }
  SetLength(ySorted, Length(y));
  ySorted := copy(y, 0, MaxInt);
  radixsort(ySorted);

  cntUnique := 1;
  SetLength(unique, cntUnique);
  currFound := ySorted[0];
  unique[0] := currFound;

  for i := 0 to Length(y) - 1 do
  begin
    if (currFound <> ySorted[i]) then
    begin
      Inc(cntUnique);
      currFound := ySorted[i];

      SetLength(unique, cntUnique);
      unique[cntUnique - 1] := currFound;
    end;
  end;

  // actual OHES encoding
  c := cntUnique;
  SetLength(res, Length(y));
  for i := 0 to Length(y) - 1 do
  begin
    SetLength(res[i], c);
    for j := 0 to c - 1 do
      res[i][j] := 0.0;
    res[i][unique[y[i]]] := 1.0;
  end;

  Result := res;
end;


end.

