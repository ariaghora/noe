unit noe.ndarr;

{$mode objfpc}{$H+}{$modeSwitch advancedRecords}

interface

uses
  Classes, Math, SysUtils, noe.types, noe.utils, strutils;

type

  TUFunc = function(v: NFloat): NFloat;
  TBFunc = function(v1, v2: NFloat): NFloat;

  { TNdArr }

  TNdArr = record
  private
    fIsContiguous: boolean;
    fShape: array of longint;
    fStrides: array of longint;
    function GetNDims: longint;
    function GetSize: longint;
  public
    Val:     TFloatVector;
    function Contiguous: TNdArr;
    function Dot(Other: TNdArr): TNdArr;
    function DumpCSV(Sep: string = ','): string;
    function GetAt(Index: array of longint): TNdArr;
    function GetShape: TIntVector;
    function Reshape(ShapeVals: array of longint): TNdArr;
    function T: TNdArr;
    function ToTensor(RequiresGrad: boolean = False): TNdArr;
    procedure Fill(v: double);
    procedure Cleanup;
    procedure SetAt(Index: array of longint; x: double);
    procedure WriteToCSV(FileName: string);
    procedure ReshapeInplace(NewShape: array of longint);
    property IsContiguous: boolean read fIsContiguous write fIsContiguous;
    property NDims: longint read GetNDims;
    property Shape: TIntVector read FShape write FShape;
    property Size: longint read GetSize;
    property Strides: TIntVector read FStrides write FStrides;
  end;

  TCallback = procedure(val: NFloat; offset:longint; idx: TIntVector; currDim: longint; var T, OutT: TNdArr);

  function CreateEmptyNdArr(Shape: array of longint): TNdArr;

  function ApplyBfunc(A, B: TNdArr; Func: TBFunc): TNdArr;
  function ApplyUfunc(A: TNdArr; Func: TUFunc): TNdArr;

  procedure Print2DArray(T: TNdArr);


implementation

procedure Print2DArray(T: TNdArr);
var
  i, j: integer;
  s: string;
begin
  Assert(T.NDims <= 2, 'Can only print a tensor with NDims = 2.');
  s := '';

  if T.NDims = 0 then
    s := s + FloatToStr(T.Val[0])
  else if T.NDims = 1 then
  begin
    for i := 0 to T.Shape[0] - 1 do
    begin
      s := s + FloatToStr(T.Val[i]);   // ENSURE CONTIGUOUS
      if i < T.Shape[0] - 1 then s := s + ' ';
    end;
  end
  else
  begin
    for i := 0 to T.Shape[0] - 1 do
    begin
      for j := 0 to T.Shape[1] - 1 do
      begin
        s := s + FloatToStr(T.Val[i * T.Shape[1] + j]);
        if j < T.Shape[1] - 1 then s := s + ' ';
      end;
      s := s + sLineBreak;
    end;
  end;
  WriteLn(s);
end;

function ApplyUfunc(A: TNdArr; Func: TUFunc): TNdArr;
var
  i: longint;
begin
  Result.ReshapeInplace(A.Shape);
  SetLength(Result.val, Length(A.val));
  for i := 0 to length(A.val) - 1 do
    Result.val[i] := func(A.val[i]);
end;

function IndexToOffset(Index, Shape, Strides: array of longint): longint;
var
  k: longint;
begin
  Result := 0;
  for k := 0 to Length(Shape) - 1 do
    Result := Result + Strides[k] * Index[k];
end;

procedure IterateTensor(T, OutT: TNdArr; Callback: TCallback);
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
        if dimTracker[j] <> res[j] then
        begin
          dimTracker[j] := res[j];

          ithDimChanged := j;
        end;
      Callback(T.Val[IndexToOffset(res, T.Shape, T.Strides)], offset, res, ithDimChanged, T, OutT);
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
  n      := Length(T.Shape);
  SetLength(res, n);
  n := Length(T.shape);
  SetLength(dimTracker, n);
  for dtIter := 0 to n - 1 do
    dimTracker[dtIter] := 0;
  iterate(0, res);
end;

procedure cbAsStrided(val: NFloat; offset: longint; idx: TIntVector;
    currDim: longint; var T, OutT: TNdArr);
begin
  OutT.Val[offset] := val;
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

function ShapeToStride(Shape: array of longint): TIntVector;
var
  k, j, prod: longint;
begin
  SetLength(Result, Length(Shape));

  for k := 0 to Length(Shape) - 1 do
  begin
    prod := 1;
    for j := k + 1 to Length(Shape) - 1 do
      prod := prod * Shape[j];
    Result[k] := prod;
  end;
end;

function AsStrided(X: TNdArr; TargetShape, Strides: array of longint): TNdArr;
var
  i: longint;
  OutStrides: TIntVector;
begin
  SetLength(Result.Val, ShapeToSize(TargetShape));

  X.ReshapeInplace(TargetShape);
  SetLength(OutStrides, Length(strides));
  for i := 0 to length(Strides) - 1 do
    OutStrides[i] := Strides[i];
  X.Strides := OutStrides;

  IterateTensor(X, Result, @cbAsStrided);
  Result.ReshapeInplace(TargetShape);
end;

function BroadcastTo(X: TNdArr; TargetShape: array of longint): TNdArr;
var
  OutShape, OutStrides: TIntVector;
  i: longint;
begin
  OutShape   := ReverseIntArr(X.Shape);
  OutStrides := ReverseIntArr(X.Strides);
  while length(OutShape) < Length(TargetShape) do
  begin
    SetLength(OutShape, Length(OutShape) + 1);
    OutShape[Length(OutShape) - 1] := 1;

    SetLength(OutStrides, Length(OutStrides) + 1);
    OutStrides[Length(OutStrides) - 1] := 0;
  end;
  OutShape   := ReverseIntArr(OutShape);
  OutStrides := ReverseIntArr(OutStrides);

  for i := 0 to Length(TargetShape) - 1 do
    if TargetShape[i] <> OutShape[i] then
      OutStrides[i] := 0;

  Result := AsStrided(X, TargetShape, OutStrides);
end;

function IsBroadcastable(A, B: TNdArr): boolean;
var
  i, violated: longint;
  revA, revB: TIntVector;
begin
  { counting the violation of broadcasting rule }
  violated := 0;
  Result   := False;
  revA     := ReverseIntArr(A.Shape);
  revB     := ReverseIntArr(B.Shape);
  for i := 0 to Math.Min(Length(A.Shape), Length(B.Shape)) - 1 do
    if (revA[i] <> revB[i]) then
      if ((revA[i] <> 1) and (revB[i] <> 1)) then
        Inc(violated);
  Result := violated = 0;
end;

function GetBroadcastDims(A, B: TNdArr): TIntVector;
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

function ApplyBfunc(A, B: TNdArr; Func: TBFunc): TNdArr;
var
  i: Longint;
  outdim: TIntVector;
begin
  { Case 1: A and B have the same shape. Perform usual element-wise operation. }
  if IntVectorEquals(A.Shape, B.Shape) then
  begin
    Result := CreateEmptyNdArr(A.Shape);
    for i := 0 to A.Size - 1 do
      Result.Val[i] := Func(A.Val[i], B.Val[i]);
  end
  else
  begin
    { General tensor broadcast bfunc }
    outdim := GetBroadcastDims(A, B);
    if not IntVectorEquals(A.Shape, outdim) then
      A := BroadcastTo(A, outdim);
    if not IntVectorEquals(B.Shape, outdim) then
      B := BroadcastTo(B, outdim);
    Result := ApplyBfunc(A, B, Func);
  end;
end;

function CreateEmptyNdArr(Shape: array of longint): TNdArr;
var
  size: LongInt;
begin
  size := ShapeToSize(Shape);
  SetLength(Result.Val, size);
  Result.ReshapeInplace(Shape);
  Result.Strides := ShapeToStride(Shape);
  Result.IsContiguous := True;
end;

function TNdArr.GetNDims: longint;
begin
  Exit(Length(Self.Shape));
end;

function TNdArr.GetSize: longint;
begin
  Exit(Length(self.Val));
end;

function TNdArr.Contiguous: TNdArr;
begin
  if Self.IsContiguous then Exit(Self)
  else
  begin
    Exit(AsStrided(Self, Self.Shape, Self.Strides));
  end;
end;

function TNdArr.Dot(Other: TNdArr): TNdArr;
begin

end;

function TNdArr.DumpCSV(Sep: string): string;
begin

end;

function TNdArr.GetAt(Index: array of longint): TNdArr;
var
  i, offset, amount: longint;
  OutShape: TIntVector;
begin
  offset := 0;
  for i := 0 to Length(Index) - 1 do
    offset := offset + Self.Strides[i] * Index[i];

  SetLength(OutShape, Length(Self.Shape) - Length(Index));
  amount := 1;
  for i := Length(Index) to Length(Self.Shape) - 1 do
  begin
    amount := amount * Self.Shape[i];
    OutShape[i - Length(Index)] := Self.Shape[i];
  end;

  SetLength(Result.Val, amount+10);
  for i := offset to offset + amount - 1 do
  begin
    Result.Val[i - offset] := Self.Val[i];
  end;

  Result.ReshapeInplace(OutShape);
end;

function TNdArr.GetShape: TIntVector;
begin
  Exit(Self.Shape);
end;

function TNdArr.Reshape(ShapeVals: array of longint): TNdArr;
var
  i: longint;
begin
  SetLength(Result.fShape, Length(ShapeVals));
  for i := 0 to Length(ShapeVals) -1 do
    Result.Shape[i] := ShapeVals[i];
  Result.Val := copy(Self.Val);
  Result.Strides := ShapeToStride(ShapeVals);
end;

function TNdArr.T: TNdArr;
begin
  Result := AsStrided(Self, ReverseIntArr(Self.Shape), ReverseIntArr(Self.Strides));
end;

function TNdArr.ToTensor(RequiresGrad: boolean): TNdArr;
begin

end;

procedure TNdArr.Fill(v: double);
var
  i: longint;
begin
  for i := 0 to Self.Size - 1 do
    self.Val[i] := v;
end;

procedure TNdArr.Cleanup;
begin
  self.val := nil;
  self.Shape := nil;
  self.Strides := nil;
end;

procedure TNdArr.SetAt(Index: array of longint; x: double);
begin

end;

procedure TNdArr.WriteToCSV(FileName: string);
begin

end;

procedure TNdArr.ReshapeInplace(NewShape: array of longint);
var
  i: longint;
begin
  SetLength(self.FShape, Length(NewShape));
  for i := 0 to Length(NewShape) - 1 do
    self.FShape[i] := NewShape[i];
  self.Strides := ShapeToStride(NewShape);
end;

end.

