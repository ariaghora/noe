unit noe.core;

{$mode objfpc}

interface

uses
  Classes, SysUtils, strutils, Math;

var
  _node_count: longint;

type
  TIntVector = array of longint;
  { TShape }
  TShape = record
    Height: longint;
    Width: longint;
  end;

  { TTensor }
  TTensor = class
    Val: array of single;
    FShape: array of longint;
  public
    function GetShape: TIntVector;
    function GetVal(Index: array of longint): single;
    procedure Reshape(ShapeVals: array of longint);
    property Shape: TIntVector read FShape;
  end;

const
  MSG_ASSERTION_DIM_MISMATCH = 'Dimension mismatch.';
  MSG_ASSERTION_INVALID_AXIS = 'Invalid axis. The value should be either 0 or 1.';

{ Operator overloading --------------------------------------------------------}
operator := (Val: single) M: TTensor;
operator +(A, B: TTensor) C: TTensor;

{ Helpers ---------------------------------------------------------------------}
function Equals(A, B: TTensor): boolean;
function IndexToOffset(Index, Shape: array of longint): longint;

{ Tensor creation ------------------------------------------------------------ }
function FullTensor(Shape: array of longint): TTensor; overload;
function FullTensor(Shape: array of longint; Val: single): TTensor; overload;
function FullTensor(Shape: array of longint; Vals: array of single): TTensor; overload;

procedure PrintTensor(T: TTensor);

implementation

uses
  noe.Math;

operator := (Val: single) M: TTensor;
begin
  M := FullTensor([1, 1], Val);
end;

operator +(A, B: TTensor) C: TTensor;
begin
  C := noe.Math.Add(A, B);
end;

function Equals(A, B: TTensor): boolean;
var
  i: longint;
begin
  Assert((A.Shape[0] = B.Shape[0]) and (A.Shape[1] = B.Shape[1]),
    MSG_ASSERTION_DIM_MISMATCH);

  Result := True;
  for i := 0 to Length(A.Val) - 1 do
    Result := Result and (A.Val[i] = B.Val[i]);
end;

function IndexToOffset(Index, Shape: array of longint): longint;
var
  i, j, d, SumRes, ProdRes: longint;
begin
  d := Length(Index);
  Assert(d = Length(Shape), 'Cannot convert index to offset with such shape');
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

{ Determine the required 1-d array size based on a tensor shape }
function ShapeToSize(Shape: array of longint): longint;
var
  i, size: longint;
begin
  size := 1;
  for i := 0 to Length(Shape) - 1 do
    size := size * shape[i];
  Result := size;
end;

{ TTensor implementations }

function TTensor.GetVal(Index: array of longint): single;
var
  Offset: longint;
begin
  Assert(length(self.Shape) = length(Index),
    'Index dimension does not match the tensor dimension');
  Offset := IndexToOffset(Index, self.Shape);
  Result := self.Val[Offset];
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

function FullTensor(Shape: array of longint; Val: single): TTensor;
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

function FullTensor(Shape: array of longint; Vals: array of single): TTensor;
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

{ iterate over a tensor }
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

    //if ithDimChanged < n - 1 then
    begin
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
  offset := 0;
  n := Length(T.Shape);

  digitMax := Math.ceil(Math.log10(Math.MaxValue(T.Val)));
  decimalPlace := 2;

  SetLength(dimTracker, n);
  for dtIter := 0 to n - 1 do
    dimTracker[dtIter] := 0;

  SetLength(res, n);
  Write(DupeString('[', n));
  iterate(0, T.GetShape, res);
  WriteLn(DupeString(']', n));
end;

end.
