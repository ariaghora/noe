{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit contains the interface for TTensor to perform multidimensional array
 operations. The dimension can be of any arbitrary nonnegative integer.

 To do:
  - implement broadcasting mechanism
  - implement a function to get string from tensor values
  - add transpose capability
}
unit noe.core;

{$mode objfpc}

interface

uses
  Classes, SysUtils, strutils, Math;

type
  TIntVector = array of longint;

  { TTensor }
  TTensor = class
    Val: array of single;
    FShape: array of longint;
  public
    function GetShape: TIntVector;
    function GetAt(Index: array of longint): TTensor;
    // Helps are welcome to implement:
    // -------------------------------
    // 1) function T(Dims: array of longint): TTensor;
    //    --> Transpose along specified dimensions
    // 2) function T: TTensor;
    //    --> Transpose along the first dimension
    procedure Reshape(ShapeVals: array of longint);
    property Shape: TIntVector read FShape;
  end;

const
  MSG_ASSERTION_DIM_MISMATCH = 'Dimension mismatch.';
  MSG_ASSERTION_INVALID_AXIS = 'Invalid axis. The value should be either 0 or 1.';
  MSG_ASSERTION_DIFFERENT_LENGTH = 'Two arrays have different length.';

{ Operator overloading --------------------------------------------------------}
operator := (Val: float) M: TTensor;
operator +(A, B: TTensor) C: TTensor;
operator -(A, B: TTensor) C: TTensor;
operator * (A, B: TTensor) C: TTensor;

{ Helpers ---------------------------------------------------------------------}

{ Check if all corresponding elements in two tensor are equal }
function Equals(A, B: TTensor): boolean;

{ Determine the offset based on given index }
function IndexToOffset(Index, Shape: array of longint): longint;

{ Determine the required 1-d array size based on a tensor shape }
function ShapeToSize(Shape: array of longint): longint;

procedure PrintTensor(T: TTensor);

{ Tensor creation ------------------------------------------------------------ }
function FullTensor(Shape: array of longint): TTensor; overload;
function FullTensor(Shape: array of longint; Val: float): TTensor; overload;
function FullTensor(Shape: array of longint; Vals: array of float): TTensor; overload;

implementation

uses
  noe.Math, noe.utils;

operator := (Val: float) M: TTensor;
begin
  M := FullTensor([1], Val);
end;

operator +(A, B: TTensor) C: TTensor;
begin
  C := noe.Math.Add(A, B);
end;

operator -(A, B: TTensor)C: TTensor;
begin
  C := noe.Math.Subtract(A, B);
end;

operator * (A, B: TTensor)C: TTensor;
begin
  C := noe.Math.Multiply(A, B);
end;

function Equals(A, B: TTensor): boolean;
var
  i: longint;
begin
  Assert((A.Shape[0] = B.Shape[0]) and (A.Shape[1] = B.Shape[1]),
    MSG_ASSERTION_DIM_MISMATCH);

  Result := (A.val = B.val);
end;

function IndexToOffset(Index, Shape: array of longint): longint;
var
  i, j, d, SumRes, ProdRes: longint;
begin
  d := Length(Index);
  Assert(d <= Length(Shape), 'Cannot convert index to offset with such shape');
  SumRes := 0;
  for i := 0 to d - 1 do
  begin
    ProdRes := 1;
    for j := i + 1 to d - 1 do
    begin
      ProdRes := ProdRes * (Shape[j]);
    end;
    SumRes := SumRes + ProdRes * Index[i];
  end;
  Result := SumRes;
end;

function ShapeToSize(Shape: array of longint): longint;
var
  i, size: longint;
begin
  size := 1;
  for i := 0 to Length(Shape) - 1 do
    size := size * shape[i];
  Result := size;
end;

function TTensor.GetAt(Index: array of longint): TTensor;
var
  i, ResultLength, offset, LIndex, LShape: longint;
  AdjustedIndex, ResultingShape: array of longint;
begin
  LIndex := Length(Index);
  LShape := Length(self.Shape);

  SetLength(AdjustedIndex, 0);      // fill with
  SetLength(AdjustedIndex, LShape); // zero

  { Suppose, the given index is [1,2] while the tensor is 4 dimensional, adjust
    the index to [1,2,0,0], i.e., fill the remaining dimension indices with
    zeros. }
  for i := 0 to LShape - 1 do
    AdjustedIndex[i] := Index[i];

  offset := IndexToOffset(AdjustedIndex, self.Shape);

  SetLength(ResultingShape, LShape - LIndex);
  for i := 0 to (LShape - LIndex) - 1 do
    ResultingShape[i] := self.Shape[i + LShape - LIndex - 1];

  Result := TTensor.Create;
  Result.Reshape(ResultingShape);

  ResultLength := 1;
  for i := LIndex to LShape - 1 do
    ResultLength := ResultLength * self.Shape[i];

  SetLength(Result.Val, ResultLength);
  for i := 0 to ResultLength - 1 do
    Result.Val[i] := self.Val[i + offset];
end;

function TTensor.GetShape: TIntVector;
begin
  Result := self.Shape;
end;

procedure TTensor.Reshape(ShapeVals: array of longint);
var
  i: longint;
begin
  SetLength(self.FShape, Length(ShapeVals));
  for i := 0 to Length(ShapeVals) - 1 do
    self.FShape[i] := ShapeVals[i];
end;

function FullTensor(Shape: array of longint): TTensor;
var
  i, size: longint;
begin
  size := ShapeToSize(Shape);
  Result := TTensor.Create;
  SetLength(Result.Val, size);
  for i := 0 to size - 1 do
    Result.Val[i] := Random;
  Result.Reshape(shape);
end;

function FullTensor(Shape: array of longint; Val: float): TTensor;
var
  i, size: longint;
begin
  size := ShapeToSize(Shape);
  Result := TTensor.Create;
  SetLength(Result.Val, size);
  for i := 0 to size - 1 do
    Result.Val[i] := Val;
  Result.Reshape(shape);
end;

function FullTensor(Shape: array of longint; Vals: array of float): TTensor;
var
  i, size: longint;
begin
  size := ShapeToSize(Shape);
  Assert(ShapeToSize(Shape) = size,
    'The values cannot be reshaped into the target shape');
  Result := TTensor.Create;
  SetLength(Result.Val, size);
  for i := 0 to size - 1 do
    Result.Val[i] := Vals[i];
  Result.Reshape(Shape);
end;

procedure PrintTensor(T: TTensor);
var
  n, offset, digitMax, decimalPlace, dtIter: longint;
  res, dimTracker: array of longint;

  procedure PPrint(res: array of longint);
  var
    i, NewlineNum, ithDimChanged: longint;
  begin
    NewlineNum := 0;

    for i := Length(res) - 1 downto 0 do
    begin
      if dimTracker[i] <> res[i] then
      begin
        dimTracker[i] := res[i];

        NewlineNum := n - i - 1;
        ithDimChanged := i; // in which dimension there is a change?
      end;
    end;

    if ithDimChanged < n - 1 then
      Write(DupeString(']', NewlineNum));

    Write(DupeString(sLineBreak, NewlineNum));

    if ithDimChanged = n - 1 then
      Write(', ');

    if ithDimChanged < n - 1 then
    begin
      Write(DupeString(' ', n - NewlineNum));
      Write(DupeString('[', NewlineNum));
    end;

    Write(T.Val[offset]: digitMax + decimalPlace + 1: decimalPlace);
  end;

  // d is dimension iterator, d=0..n-1
  procedure iterate(d: longint; shape, res: array of longint);
  var
    i: longint;
  begin
    if d >= n then
    begin
      PPrint(res);
      Inc(offset);
      exit;
    end;

    for i := 0 to shape[d] - 1 do
    begin
      res[d] := i;
      iterate(d + 1, shape, res);
    end;
  end;

begin
  digitMax := Math.ceil(Math.log10(Math.MaxValue(T.Val) + 0.01));
  decimalPlace := 2;

  if Length(T.Val) = 1 then { it is a scalar }
    writeln(T.Val[0]: digitMax + decimalPlace + 1: decimalPlace)
  else { it is a higher rank tensor }
  begin
    offset := 0;
    n := Length(T.Shape);

    SetLength(dimTracker, n);
    for dtIter := 0 to n - 1 do
      dimTracker[dtIter] := 0;

    SetLength(res, n);
    Write(DupeString('[', n));
    iterate(0, T.GetShape, res);
    WriteLn(DupeString(']', n));
  end;
end;

end.
