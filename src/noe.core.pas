unit noe.core;

{$mode objfpc}

interface

uses
  Classes, SysUtils;

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

{ Create a matrix with randomized float values }
function FullFloat(Arr: array of single): TTensor; overload;
function FullFloat(Height, Width: longint): TTensor; overload;
function FullFloat(Height, Width: longint; Val: single): TTensor; overload;
function FullTensor(Shape: array of longint): TTensor; overload;
function FullTensor(Shape: array of longint; Val: single): TTensor; overload;
function FullTensor(Shape, Vals: array of longint): TTensor; overload;

procedure PrintTensor(M: TTensor; Preamble: string); overload;
procedure PrintTensor(M: TTensor);

implementation

uses
  noe.Math;

operator := (Val: single) M: TTensor;
begin
  M := FullFloat(1, 1, Val);
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

function FullFloat(Arr: array of single): TTensor;
begin
  Result.Val := @Arr;
  SetLength(Result.Val, Length(Arr));
  Result.Reshape([1, Length(Arr)]);
end;

{ Utility functions }
function FullFloat(Height, Width: longint): TTensor;
var
  i, length: longint;
begin
  Result := TTensor.Create;
  length := Height * Width;
  SetLength(Result.Val, length);
  for i := 0 to length - 1 do
  begin
    Result.val[i] := Random;
  end;
  Result.Reshape([Height, Width]);
end;

function FullFloat(Height, Width: longint; Val: single): TTensor;
var
  i, length: longint;
begin
  Result := TTensor.Create;
  length := Height * Width;
  SetLength(Result.Val, length);
  for i := 0 to length - 1 do
  begin
    Result.val[i] := Val;
  end;
  Result.Reshape([Height, Width]);
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

function FullTensor(Shape, Vals: array of longint): TTensor;
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


procedure PrintTensor(M: TTensor; Preamble: string);
var
  i, j: longint;
begin
  writeln(Preamble, ':', sLineBreak);
  for i := 0 to M.Shape[0] - 1 do
  begin
    for j := 0 to M.Shape[1] - 1 do
    begin
      Write(M.Val[i * M.Shape[1] + j]: 8: 2);
      if j < (M.Shape[1] - 1) then
        Write(' ');
    end;
    WriteLn();
  end;
  WriteLn();
end;

procedure PrintTensor(M: TTensor);
var
  preamble: string;
begin
  preamble := 'shape=' + IntToStr(M.Shape[0]) + 'x' + IntToStr(M.Shape[1]);
  preamble := preamble + ' (TMatrix)';
  PrintTensor(M, preamble);
end;

end.
