{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit contains the interface for TTensor to perform multidimensional array
 operations. The dimension can be of any arbitrary nonnegative integer.

 To do:
  - implement broadcasting mechanism
  - implement a function to get string from tensor values
}
unit noe.core;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, strutils, Math;

type
  TIntVector   = array of longint;
  TFloatVector = array of double;

  { TTensor }
  TTensor = class
    Val:    TFloatVector;
    FShape: array of longint;
  public
    function DumpCSV(Sep: string = ','): string;
    function GetShape: TIntVector;
    function GetAt(Index: array of longint): TTensor;
    { transpose for matrix, reverse index for tensors }
    function T: TTensor;
    procedure Reshape(ShapeVals: array of longint);
    property Shape: TIntVector read FShape;
  end;

  PTensor    = ^TTensor;
  TTensorArr = array of TTensor;

  { A proxy of a tensor broadcasted to a specific dimension. The proxies are
    created whenever a broadcasting is needed to prevent the creation of exact
    copy of a tensor }

  { TTensorProxy }

  TTensorProxy = class
    FRef:      TTensor;
    { Adjusted reference dimension (prepending the dimension when needed) }
    FRefShape: TIntVector;
    FIndices:  TIntVector;
    FTargetShape: TIntVector;
  public
    constructor Create(var ref: TTensor; targetShape: TIntVector);

    { Get the value pointing to FRef at the (mapped) offset.
      TODO in the future, precompute the mapping in the constructor. Or not? }
    function GetValByOffset(offset: longint): double;
    function MapOffset(offset: longint): longint;
    procedure GenerateIndex;
    property Val[i: longint]: double read GetValByOffset;
  end;

  TBroadcastResult = record
    A: TTensorProxy;
    B: TTensorProxy;
    broadcastShape: TIntVector;
  end;

  TConfig = record
    debug:   boolean;
    useBLAS: boolean;
    backend: string;
    BLASFileName: string;
  end;

  TCallback = procedure(val: float; idx: TIntVector; currDim: longint; var T: TTensor);

const
  {$I config}
  MSG_ASSERTION_DIM_MISMATCH     = 'Dimension mismatch.';
  MSG_ASSERTION_INVALID_AXIS     = 'Invalid axis. The value should be either 0 or 1.';
  MSG_ASSERTION_DIFFERENT_LENGTH = 'Two arrays have different length.';
  MSG_ASSERTION_RANK_2_TENSORS_ONLY = 'This function can be used only on rank-2 tensors';

var
  NoeConfig: TConfig;

{ Operator overloading --------------------------------------------------------}
operator := (Val: float) M: TTensor;
operator +(A, B: TTensor) C: TTensor;
operator -(A: TTensor) B: TTensor;
operator -(A, B: TTensor) C: TTensor;
operator / (A, B: TTensor) C: TTensor;
operator * (A, B: TTensor) C: TTensor;
operator ** (A: TTensor; expo: double) B: TTensor; inline;
operator ** (A, B: TTensor) C: TTensor; inline;


{ Helpers ---------------------------------------------------------------------}

{ Check if all corresponding elements in two tensor are equal }
function Equals(A, B: TTensor): boolean;

function DimsToLetter(dims: array of longint): string;

{ Determine the offset based on given multidimensional index }
function IndexToOffset(Index, Shape: array of longint): longint;

{ Determine the multidimensional index based on given offset }
function OffsetToIndex(offset: longint; Shape: array of longint): TIntVector;

{ Determine the required 1-d array size based on a tensor shape }
function ShapeToSize(Shape: array of longint): longint;

{ Helpers API for matrix (rank-2 tensor) --------------------------------------}
function GetRange(T: TTensor; RowIndex, ColumnIndex, Height, Width: longint): TTensor;
function GetColumn(T: TTensor; ColumnIndex: longint): TTensor;
function GetRow(T: TTensor; RowIndex: longint): TTensor;

{ Broadcasting ----------------------------------------------------------------}

{ Check if two tensors are broadcasatable }
function IsBroadcastable(A, B: TTensor): boolean;
function GetBroadcastDims(A, B: TTensor): TIntVector;
function Broadcast(A, B: TTensor): TBroadcastResult;

procedure PrintTensor(T: TTensor);
procedure IterateTensor(T: TTensor; Callback: TCallback);

{ Tensor creation ------------------------------------------------------------ }
function CopyTensor(A: TTensor): TTensor;
function FullTensor(Shape: array of longint): TTensor; overload;
function FullTensor(Shape: array of longint; Val: float): TTensor; overload;
function FullTensor(Shape: array of longint; Vals: array of float): TTensor; overload;
function Zeros(Shape: array of longint): TTensor;
function Ones(Shape: array of longint): TTensor;

{ Generates an array of float within range of (0, n] }
function Range(start, stop, step: double): TTensor;
function Range(start, stop: double): TTensor; overload;
function Range(n: longint): TTensor; overload;
//function LinSpace():

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

operator -(A: TTensor)B: TTensor;
var
  i: longint;
begin
  B := CopyTensor(A);
  for i := 0 to Length(B.val) - 1 do
    B.val[i] := -A.val[i];
end;

operator -(A, B: TTensor)C: TTensor;
begin
  C := noe.Math.Subtract(A, B);
end;

operator / (A, B: TTensor)C: TTensor;
begin
  C := noe.Math.Divide(A, B);
end;

operator * (A, B: TTensor)C: TTensor;
begin
  C := noe.Math.Multiply(A, B);
end;

operator ** (A: TTensor; expo: double)B: TTensor;
begin
  B := Power(A, expo);
end;

operator ** (A, B: TTensor)C: TTensor;
begin
  C := Power(A, B);
end;

function Equals(A, B: TTensor): boolean;
begin
  Assert((A.Shape[0] = B.Shape[0]) and (A.Shape[1] = B.Shape[1]),
    MSG_ASSERTION_DIM_MISMATCH);

  Result := (A.val = B.val);
end;

function DimsToLetter(dims: array of longint): string;
var
  alphabet: string = 'abcdefghijklmnopqrstuvwxyz';
begin
  Result := Copy(alphabet, 1, Length(dims));
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

function OffsetToIndex(offset: longint; Shape: array of longint): TIntVector;
var
  dim, cnt: longint;
begin
  SetLength(Result, Length(Shape));
  cnt := 0;
  for dim in ReverseIntArr(Shape) do
  begin
    Result[cnt] := offset mod dim;
    offset := offset div dim;
    cnt := cnt + 1;
  end;

  Result := ReverseIntArr(Result);
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

{ TTensorProxy }

constructor TTensorProxy.Create(var ref: TTensor; targetShape: TIntVector);
var
  i, nExtraDim: longint;
begin
  FRef := ref;
  FTargetShape := targetShape;

  nExtraDim := Length(targetShape) - Length(ref.Shape);
  SetLength(FRefShape, Length(ref.Shape) + nExtraDim);
  for i := 0 to Length(FRefShape) - 1 do
    FRefShape[i] := 1;
  for i := 0 to length(ref.Shape) - 1 do
    FRefShape[i] := ReverseIntArr(ref.Shape)[i];
  FRefShape := ReverseIntArr(FRefShape);
end;

function TTensorProxy.GetValByOffset(offset: longint): double;
var
  expectedIndex, mappedIndex: TIntVector;
  i: longint;
begin
  expectedIndex := OffsetToIndex(offset, FTargetShape);

  SetLength(mappedIndex, Length(expectedIndex));

  { map expected index to the proper index }
  for i := 0 to Length(expectedIndex) - 1 do
  begin
    if expectedIndex[i] > FRefShape[i] - 1 then
    begin
      mappedIndex[i] := 0;
    end
    else
      mappedIndex[i] := expectedIndex[i];
  end;

  Result := FRef.Val[IndexToOffset(mappedIndex, FRefShape)];
end;

function TTensorProxy.MapOffset(offset: longint): longint;
var
  expectedIndex, mappedIndex: TIntVector;
  i: longint;
begin
  expectedIndex := OffsetToIndex(offset, FTargetShape);

  SetLength(mappedIndex, Length(expectedIndex));

  { map expected index to the proper index }
  for i := 0 to Length(expectedIndex) - 1 do
  begin
    if expectedIndex[i] > FRefShape[i] - 1 then
    begin
      mappedIndex[i] := 0;
    end
    else
      mappedIndex[i] := expectedIndex[i];
  end;

  Result := IndexToOffset(mappedIndex, FRefShape);
end;

procedure TTensorProxy.GenerateIndex;
var
  i: longint;
begin
  SetLength(FIndices, Length(FRef.Val));
  for i := 0 to Length(FRef.Val) - 1 do
    FIndices[i] := MapOffset(i);
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

function TTensor.T: TTensor;
begin
  Result := noe.Math.Transpose(Self);
end;

function TTensor.DumpCSV(Sep: string = ','): string;
var
  i, j: integer;
begin
  Assert(Length(self.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := '';
  for i := 0 to self.Shape[0] - 1 do
  begin
    for j := 0 to self.Shape[1] - 1 do
    begin
      Result := Result + FloatToStr(self.val[i * self.Shape[1] + j]);
      if j < self.Shape[1] - 1 then
        Result := Result + sep;
    end;
    if i < self.Shape[0] - 1 then
      Result := Result + LineEnding;
  end;
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

function CopyTensor(A: TTensor): TTensor;
begin
  Result := TTensor.Create;
  Result.val := copy(A.val);
  Result.Reshape(A.Shape);
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

function Zeros(Shape: array of longint): TTensor;
begin
  Result := FullTensor(Shape, 0);
end;

function Ones(Shape: array of longint): TTensor;
begin
  Result := FullTensor(Shape, 1.0);
end;

function Range(start, stop, step: double): TTensor;
var
  i: double;
  offset: longint;
begin
  Result := TTensor.Create;
  Result.Reshape([floor((stop - start) / step)]);
  SetLength(Result.Val, floor((stop - start) / step));

  i := start;
  offset := 0;
  while i < stop do
  begin
    Result.Val[offset] := i;
    i := i + step;
    Inc(offset);
  end;
end;

function Range(start, stop: double): TTensor;
begin
  Result := Range(start, stop, 1);
end;

function Range(n: longint): TTensor;
begin
  Result := Range(0, n, 1);
end;

function GetRange(T: TTensor; RowIndex, ColumnIndex, Height, Width: longint): TTensor;
var
  i, j, offset: longint;
begin
  Assert(Length(T.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := TTensor.Create;
  Result.Reshape([Height, Width]);

  SetLength(Result.Val, Height * Width);
  offset := 0;
  for i := RowIndex to RowIndex + Height - 1 do
  begin
    for j := ColumnIndex to ColumnIndex + Width - 1 do
    begin
      Result.Val[offset] := T.Val[i * T.Shape[1] + j];
      Inc(offset);
    end;
  end;
end;

function GetColumn(T: TTensor; ColumnIndex: longint): TTensor;
begin
  Result := GetRange(T, 0, ColumnIndex, T.Shape[1], 1);
end;

function GetRow(T: TTensor; RowIndex: longint): TTensor;
begin
  Result := GetRange(T, RowIndex, 0, 1, T.Shape[0]);
end;

procedure IterateTensor(T: TTensor; Callback: TCallback);
var
  n, offset, ithDimChanged, dtIter: longint;
  res, dimTracker: TIntVector;

  procedure iterate(d: longint; res: TIntVector);
  var
    i, j: longint;
  begin
    if d >= n then
    begin
      for j := Length(res) - 1 downto 0 do
      begin
        if dimTracker[j] <> res[j] then
        begin
          dimTracker[j] := res[j];

          //NewlineNum := n - j - 1;
          ithDimChanged := j; // in which dimension there is a change?
        end;
      end;

      Callback(T.Val[offset], res, ithDimChanged, T);
      Inc(offset);
      exit;
    end;

    for i := 0 to T.shape[d] - 1 do
    begin
      res[d] := i;
      iterate(d + 1, res);
    end;
  end;

begin
  offset := 0;
  n := Length(T.Shape);
  SetLength(res, n);
  n := Length(T.shape);
  SetLength(dimTracker, n);
  for dtIter := 0 to n - 1 do
    dimTracker[dtIter] := 0;
  iterate(0, res);
end;

function IsBroadcastable(A, B: TTensor): boolean;
var
  i, violated: longint;
  revA, revB: TIntVector;
begin
  { counting the violation of broadcasting rule }
  violated := 0;
  Result := False;
  revA := ReverseIntArr(A.Shape);
  revB := ReverseIntArr(B.Shape);
  for i := 0 to Min(Length(A.Shape), Length(B.Shape)) - 1 do
  begin
    if (revA[i] <> revB[i]) then
      if ((revA[i] <> 1) and (revB[i] <> 1)) then
      begin
        Inc(violated);
      end;
  end;
  Result := violated = 0;
end;

function GetBroadcastDims(A, B: TTensor): TIntVector;
var
  i, finalDimSize: longint;
  revA, revB: TIntVector;
begin
  Assert(IsBroadcastable(A, B), 'A and B cannot be broadcasted');
  finalDimSize := Max(Length(A.Shape), Length(B.Shape));

  SetLength(Result, finalDimSize);
  SetLength(revA, finalDimSize);
  SetLength(revB, finalDimSize);
  for i := 0 to Length(Result) - 1 do
  begin
    revA[i] := 1;
    revB[i] := 1;
  end;

  for i := 0 to length(A.Shape) - 1 do
    revA[i] := ReverseIntArr(A.Shape)[i];

  for i := 0 to Length(B.Shape) - 1 do
    revB[i] := ReverseIntArr(B.Shape)[i];

  revA := ReverseIntArr(revA);
  revB := ReverseIntArr(revB);
  for i := 0 to Max(Length(A.Shape), Length(B.Shape)) - 1 do
    Result[i] := max(revA[i], revB[i]);
end;

function Broadcast(A, B: TTensor): TBroadcastResult;
var
  outDim: TIntVector;
begin
  outDim := GetBroadcastDims(A, B);
  Result.A := TTensorProxy.Create(A, outDim);
  Result.B := TTensorProxy.Create(B, outDim);
  Result.broadcastShape := copy(outDim);

end;

procedure PrintTensor(T: TTensor);
var
  n, offset, digitMax, decimalPlace, dtIter: longint;
  res, dimTracker: array of longint;
  outstr: string = '';

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
      outstr := outstr + (DupeString(']', NewlineNum));

    outstr := outstr + (DupeString(sLineBreak, NewlineNum));

    if ithDimChanged = n - 1 then
      outstr := outstr + (', ');

    if ithDimChanged < n - 1 then
    begin
      outstr := outstr + (DupeString(' ', n - NewlineNum));
      outstr := outstr + (DupeString('[', NewlineNum));
    end;

    outstr := outstr + FloatToStrF(T.Val[offset], ffFixed, digitMax, decimalPlace);
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

  function MaxAbs(arr: array of double): double;
  var
    i: double;
  begin
    Result := 0;
    for i in arr do
      if i > Result then
        Result := i;
  end;

begin
  digitMax := Math.ceil(Math.log10(abs(MaxAbs(T.Val)) + 0.01));
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
    outstr := outstr + (DupeString('[', n));
    iterate(0, T.GetShape, res);
    outstr := outstr + (DupeString(']', n));
    outstr := outstr + sLineBreak;

    Write(outstr);
  end;
end;

initialization
  NoeConfig.debug := True;
  NoeConfig.backend := 'BLAS';
  NoeConfig.BLASFileName := BLAS_FILENAME;
  NoeConfig.useBLAS := True;
end.
