{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit contains the interface for TTensor to perform multidimensional array
 operations. The dimension can be of any arbitrary nonnegative integer.
}
unit noe;

{$mode objfpc}{$H+}{$modeSwitch advancedRecords}

interface

uses
  Classes, Math, strutils, SysUtils, fgl;

type
  NFloat = double;

  TIntVector   = array of longint;
  TFloatVector = array of NFloat;
  TVariable    = class;

  { TTensor }
  TTensor = record
  private
    FShape: array of longint;
    FStrides: array of longint;
    function GetNDims: longint;
    function GetSize: longint;
  public
    Val:    TFloatVector;
    function Dot(Other: TTensor): TTensor;
    function DumpCSV(Sep: string = ','): string;
    function GetAt(i: longint): double;
    function GetAt(i, j: longint): double;
    function GetAt(Index: array of longint): TTensor;
    function GetShape: TIntVector;
    function Reshape(ShapeVals: array of longint): TTensor;
    function T: TTensor;
    function ToVariable(RequiresGrad: boolean = False): TVariable;
    procedure Fill(v: double);
    procedure Free;
    procedure SetAt(i: longint; x: double);
    procedure SetAt(i, j: longint; x: double);
    procedure SetAt(Index: array of longint; x: double);
    procedure WriteToCSV(FileName: string);
    procedure ReshapeInplace(ShapeVals: array of longint);
    property NDims: longint read GetNDims;
    property Shape: TIntVector read FShape write FShape;
    property Size: longint read GetSize;
    property Strides: TIntVector read FStrides write FStrides;
  end;

  TTensorHelper = record helper for TTensor
    const Default: TTensor = (FShape:nil; FStrides: nil; val: nil);
  end;

  PTensor    = ^TTensor;
  TTensorArr = array of TTensor;

  TConfig = record
    debug:   boolean;
    useBLAS: boolean;
    backend: string;
    BLASFileName: string;
  end;

  TCallback = procedure(val: NFloat; offset:longint; idx: TIntVector; currDim: longint; var T, OutT: TTensor);

  { The wrapper of TTensor that also acts as a single node in a computaional graph }
  PVariable = ^TVariable;

  TVariableArr  = array of TVariable;
  PVariableArr  = array of ^TVariable;
  TBackwardFunc = procedure(arr: TVariableArr; ADy: TTensor);

  { TVariable }

  TVariable = class
    Prev: TVariableArr;
  private
    FTensor: TTensor;
    FGrad:   TTensor;
    FID:     longint;
    FIsLeaf: boolean;
    FRequiresGrad: boolean;
    FBackwardFunc: TBackwardFunc;
    FName:   string;
    FTrackingID: string;
    function GetNDims: longint;
    function GetShape: TIntVector;
    function GetSize: longint;
    procedure SetData(AValue: TTensor);
    procedure SetRequiresGrad(AValue: boolean);
  public
    constructor Create; overload;
    constructor Create(AName: string); overload;
    constructor Create(ATensor: TTensor); overload;
    constructor Create(ATensor: TTensor; AName: string); overload;
    constructor Create(ATensor: TTensor; AName: string;
      ABackwardFunc: TBackwardFunc); overload;
    constructor Create(ATensor: TTensor; AName: string;
      ABackwardFunc: TBackwardFunc; AIsLeaf: boolean); overload;
    destructor Cleanup;
    procedure AddPrev(AVariable: TVariable);
    procedure AddPrev(arr: array of TVariable);
    procedure Backpropagate;
    procedure FreeData;
    procedure FreeGrad;
    procedure ZeroGrad;
    property BackwardFunc: TBackwardFunc read FBackwardFunc write FBackwardFunc;
    property Data: TTensor read FTensor write SetData;
    property Grad: TTensor read FGrad write FGrad;
    property ID: longint read FID write FID;
    property IsLeaf: boolean read FIsLeaf write FIsLeaf;
    property Name: string read FName write FName;
    property NDims: longint read GetNDims;
    property RequiresGrad: boolean read FRequiresGrad write SetRequiresGrad;
    property Shape: TIntVector read GetShape;
    property Size: longint read GetSize;
    property TrackingID: string read FTrackingID write FTrackingID;

    { Math helpers }
    function Dot(Other: TVariable): TVariable;
  end;

  { TNodeTracker }
  TVariableList = specialize TFPGList<TVariable>;

  TNodeTracker = record
    Items: TVariableArr;
    NodeSpace: TVariableList;
    procedure Add(V: TVariable);
    procedure ClearUnusedNodes(root: TVariable);
    function FindByTrackingID(TrackingID: string): longint;
  end;

const
  {$I config}
  MSG_ASSERTION_DIM_MISMATCH     = 'Dimension mismatch.';
  MSG_ASSERTION_INVALID_AXIS     = 'Invalid axis. The value should be either 0 or 1.';
  MSG_ASSERTION_DIFFERENT_LENGTH = 'Two arrays have different length.';
  MSG_ASSERTION_RANK_2_TENSORS_ONLY = 'This function can be used only on rank-2 tensors';
  MSG_ASSERTION_RANK_1_TENSORS_ONLY = 'This function can be used only on rank-1 tensors';

  EPS_TOL = 1e-5;

var
  NoeConfig: TConfig;
  GLOBAL_NODE_COUNT: integer;
  GLOBAL_SKIP_GRAD:  boolean;
  GlobalNodeTracker: TNodeTracker;

{ Operator overloading --------------------------------------------------------}
operator := (Val: float) M: TTensor;
operator := (Val: double) V: TVariable;
operator := (Val: TTensor) V: TVariable;
operator +(A, B: TTensor) C: TTensor;
operator +(A, B: TVariable) C: TVariable;
operator -(A: TTensor) B: TTensor;
operator -(A: TVariable) B: TVariable;
operator -(A, B: TTensor) C: TTensor;
operator -(A, B: TVariable) C: TVariable;
operator / (A, B: TTensor) C: TTensor;
operator / (A, B: TVariable) C: TVariable;
operator * (A, B: TTensor) C: TTensor;
operator * (A, B: TVariable) C: TVariable;
operator ** (A: TTensor; expo: double) B: TTensor;
operator ** (A, B: TTensor) C: TTensor;
operator in (T: TVariable; arr: array of TVariable) b: boolean;
operator explicit (Val: TVariable) M: TTensor;
operator explicit (Val: TTensor) M: TVariable;


{ Helpers ---------------------------------------------------------------------}

function ArgMax(V: TFloatVector): longint;

{ Check if all corresponding elements in two tensor are equal }
function Equals(A, B: TTensor): boolean;

function DimsToLetter(dims: array of longint): string;

{ Determine the offset based on given multidimensional index }
function IndexToOffset(Index, Shape: array of longint): longint;
function IndexToOffset(Index, Shape, Strides: array of longint): longint;
{ Determine the multidimensional index based on given offset }
function OffsetToIndex(offset: longint; Shape: array of longint): TIntVector;
{ Determine the required 1-d array size based on a tensor shape }
function ShapeToSize(Shape: array of longint): longint;
function ShapeToStride(Shape: array of longint): TIntVector;
function Squeeze(T: TTensor): TTensor;

{ Helpers API for matrix (rank-2 tensor) --------------------------------------}
function GetRange(T: TTensor; RowIndex, ColumnIndex, Height, Width: longint): TTensor;
function GetRange(T: TVariable; RowIndex, ColumnIndex, Height, Width: longint): TTensor;
function GetColumn(T: TTensor; ColumnIndex: longint; KeepDims: boolean = false): TTensor;
function GetColumnRange(T: TTensor; ColumnIndex, Amount: longint): TTensor;
function GetRow(T: TTensor; RowIndex: longint; KeepDims: boolean = false): TTensor;
function GetRowRange(T: TTensor; RowIndex, Amount: longint): TTensor;
function VFlip(T: TTensor): TTensor;

{ Broadcasting ----------------------------------------------------------------}

function AsStrided(X: TTensor; TargetShape, Strides: array of longint): TTensor;
function BroadcastTo(X: TTensor; TargetShape: array of longint): TTensor;

{ Check if two tensors are broadcasatable }
function IsBroadcastable(A, B: TTensor): boolean;
function GetBroadcastDims(A, B: TTensor): TIntVector;

{ Tile column tensor A n times to the right }
{ HACK: it works, but certainly can be improved }
function TileColumn(A: TTensor; n: longint): TTensor;

{ Tile row tensor A n times to bottom }
{ HACK: it works, but certainly can be improved }
function TileRow(A: TTensor; n: longint): TTensor;

procedure PrintTensor(T: TTensor);
procedure PrintTensor(V: TVariable);
procedure IterateTensor(T, OutT: TTensor; Callback: TCallback);

{ Tensor creation ------------------------------------------------------------ }
function CopyTensor(A: TTensor): TTensor;
function CreateEmptyTensor(Shape: array of longint): TTensor;
function CreateTensor(Shape: array of longint; Val: NFloat): TTensor; overload;
function CreateTensor(Shape: array of longint; Vals: array of NFloat): TTensor; overload;
function Ones(Shape: array of longint): TTensor;
function RandomTensorNormal(Shape: array of longint): TTensor;
function RandomTensorBinomial(Shape: array of longint; p: double): TTensor;
function ReadCSV(fileName: string; NRowSkip: longint = 0): TTensor;
function Zeros(Shape: array of longint): TTensor;


{ Generates an array of float within range of (0, n] }
function Range(start, stop, step: double): TTensor;
function Range(start, stop: double): TTensor;
function Range(n: longint): TTensor;

{ Computational graph ---------------------------------------------------------}
function TopologicalSort(T: TVariable): TVariableArr;
procedure BackwardGraph(const T: TVariable);
procedure ClearIntermediaryNodes;
procedure SetRequiresGrad(arr: array of TVariable; val: boolean);
procedure ZeroGradGraph(const T: TVariable);

{ Auxilaries ------------------------------------------------------------------}
procedure Cleanup;

implementation

uses
  noe.Math, noe.utils;

operator := (Val: float) M: TTensor;
begin
  M := CreateTensor([1], Val);
end;

operator := (Val: TTensor)V: TVariable;
begin
  V := TVariable.Create(Val);
end;

operator +(A, B: TTensor) C: TTensor;
begin
  C := Add(A, B);
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

operator / (A, B: TVariable)C: TVariable;
begin
  C := Divide(A, B);
end;

operator * (A, B: TTensor)C: TTensor;
begin
  C := Multiply(A, B);
end;

operator ** (A: TTensor; expo: double)B: TTensor;
begin
  B := Power(A, expo);
end;

operator ** (A, B: TTensor)C: TTensor;
begin
  C := Power(A, B);
end;

operator := (Val: double)V: TVariable;
begin
  V := TVariable.Create(Val);
  V.RequiresGrad := False;

  { all constants are given id 1 }
  //V.ID := -1;
end;

operator +(A, B: TVariable)C: TVariable;
begin
  C := Add(A, B);
end;

operator -(A, B: TVariable)C: TVariable;
begin
  C := Subtract(A, B);
end;

operator -(A: TVariable)B: TVariable;
begin
  B := Negate(A);
end;

operator * (A, B: TVariable)C: TVariable;
begin
  C := Multiply(A, B);
end;

operator in(T: TVariable; arr: array of TVariable)b: boolean;
var
  Tmp: TVariable;
begin
  result := false;
  for Tmp in arr do
    if T.GetHashCode = Tmp.GetHashCode then
    begin
      result := true;
      exit;
    end;
end;

operator explicit(Val: TVariable)M: TTensor;
begin
  M := Val.Data;
end;

operator explicit(Val: TTensor)M: TVariable;
begin
  M := Val.ToVariable(False);
end;

function ArgMax(V: TFloatVector): longint;
var
  i: longint;
  CurMax: double;
begin
  CurMax := -Infinity;
  for i := 0 to Length(V) - 1 do
    if V[i] > CurMax then
    begin
      CurMax := V[i];
      Result := i;
    end;
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
      ProdRes := ProdRes * (Shape[j]);
    SumRes    := SumRes + ProdRes * Index[i];
  end;
  Result := SumRes;
end;

function IndexToOffset(Index, Shape, Strides: array of longint): longint;
var
  k: longint;
begin
  Result := 0;
  for k := 0 to Length(Shape) - 1 do
    Result := Result + Strides[k] * Index[k];
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

{ TNodeTracker }

procedure TNodeTracker.Add(V: TVariable);
begin
  SetLength(Self.Items, Length(self.Items) + 1);
  Self.Items[Length(self.Items) - 1] := V;
end;

procedure TNodeTracker.ClearUnusedNodes(root: TVariable);
var
  CurrentGraphNodes: TVariableArr;
  TobeRemoved: TVariableList;
  v, w, x, y: TVariable;
  i: longint;
begin
  CurrentGraphNodes := TopologicalSort(root);

  TobeRemoved := TVariableList.Create;
  for v in NodeSpace do
    if not(v in CurrentGraphNodes) and not(v.IsLeaf) then
      TobeRemoved.Add(v);

  for w in TobeRemoved do
  begin
    w.FreeData;
    w.FreeGrad;
    Finalize(w.FName);
    NodeSpace.Remove(w);
    // for now idk why cannot destroy :(
    //w.Destroy;
  end;

  FreeAndNil(TobeRemoved);
end;

function TNodeTracker.FindByTrackingID(TrackingID: string): longint;
var
  i: longint;
begin
  Result := -1;
  for i:=0 to Length(self.Items) - 1 do
  begin
    if self.Items[i].TrackingID = TrackingID then
      exit(i);
  end;
end;

function TTensor.GetAt(Index: array of longint): TTensor;
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

procedure TTensor.SetAt(i: longint; x: double);
begin
  assert(self.NDims = 1, MSG_ASSERTION_RANK_1_TENSORS_ONLY);
  self.Val[IndexToOffset([i], self.Shape)] := x;
end;

procedure TTensor.SetAt(i, j: longint; x: double);
begin
  assert(self.NDims = 2, MSG_ASSERTION_RANK_1_TENSORS_ONLY);
  self.Val[IndexToOffset([i, j], Self.Shape)] := x;
end;

procedure TTensor.SetAt(Index: array of longint; x: double);
begin
  self.Val[IndexToOffset(Index, Self.Shape)] := x;
end;

procedure TTensor.WriteToCSV(FileName: string);
var
  F: TextFile;
begin
  AssignFile(F, FileName);
  try
    ReWrite(F);
    Write(F, self.DumpCSV());
  finally
    CloseFile(F);
  end;
end;

function TTensor.T: TTensor;
begin
  Result := noe.Math.Transpose(Self);
end;

function TTensor.ToVariable(RequiresGrad: boolean): TVariable;
begin
  Result := TVariable.Create(self);
  Result.RequiresGrad := RequiresGrad;
end;

procedure TTensor.Fill(v: double);
var
  i: longint;
begin
  for i := 0 to Length(self.Val) - 1 do
    self.Val[i] := v;
end;

procedure TTensor.Free;
begin
  SetLength(self.Val, 0);
  SetLength(self.FShape, 0);
  SetLength(self.FStrides, 0);
end;

function TTensor.GetAt(i: longint): double;
begin
  assert(self.NDims = 1, MSG_ASSERTION_RANK_1_TENSORS_ONLY);
  Result := self.GetAt([i]).Val[0];
end;

function TTensor.GetAt(i, j: longint): double;
begin
  assert(self.NDims = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := self.GetAt([i, j]).Val[0];
end;

function TTensor.GetNDims: longint;
begin
  Result := length(self.Shape);
end;

function TTensor.GetSize: longint;
begin
  Result := Length(self.Val);
end;

function TTensor.DumpCSV(Sep: string = ','): string;
var
  i, j: integer;
begin
  Assert(Length(self.Shape) <= 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
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

function TTensor.Reshape(ShapeVals: array of longint): TTensor;
var
  i: longint;
begin
  Result := CopyTensor(self);
  SetLength(Result.FShape, Length(ShapeVals));
  for i :=0 to Length(ShapeVals) - 1 do
    Result.FShape[i] := ShapeVals[i];
  Result.Strides := ShapeToStride(ShapeVals);
end;

procedure TTensor.ReshapeInplace(ShapeVals: array of longint);
var
  i: longint;
begin
  SetLength(self.FShape, Length(ShapeVals));
  for i := 0 to Length(ShapeVals) - 1 do
    self.FShape[i] := ShapeVals[i];
  self.Strides := ShapeToStride(ShapeVals);
end;

function TTensor.Dot(Other: TTensor): TTensor;
begin
  Assert((Self.NDims <= 2) and (Other.NDims <= 2), MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := MatMul(self, Other);
end;

{ TVariable }
procedure TVariable.SetData(AValue: TTensor);
begin
  FTensor := AValue;
end;

procedure TVariable.SetRequiresGrad(AValue: boolean);
begin
  if FRequiresGrad=AValue then Exit;
  FRequiresGrad:=AValue;
  self.Grad := Zeros(self.Shape);
end;

function TVariable.GetShape: TIntVector;
begin
  Result := self.Data.Shape;
end;

function TVariable.GetSize: longint;
begin
  Result := Self.Data.Size;
end;

function TVariable.GetNDims: longint;
begin
  Result := Length(self.Shape);
end;

constructor TVariable.Create;
var
  T: TTensor;
begin
  self.Create(T, '', nil, True);
  //self.FID := -2;
end;

constructor TVariable.Create(AName: string);
var
  T: TTensor;
begin
  self.Create(T, AName, nil, True);
end;

constructor TVariable.Create(ATensor: TTensor);
begin
  self.Create(ATensor, '', nil, True);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string);
begin
  self.Create(ATensor, AName, nil, True);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string;
  ABackwardFunc: TBackwardFunc);
begin
  { it has a Backpropagate function, so it must be non-leaf }
  self.Create(ATensor, AName, ABackwardFunc, False);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string;
  ABackwardFunc: TBackwardFunc; AIsLeaf: boolean);
begin
  self.Data   := ATensor;
  self.Name   := AName;
  self.BackwardFunc := ABackwardFunc;
  self.IsLeaf := AIsLeaf;

  { always true on creation unless specified otherwise }
  self.RequiresGrad := False;

  self.ZeroGrad;

  { we need to keep track every single node created, e.g., for later removal }
  GlobalNodeTracker.NodeSpace.Add(self);

  self.FID := GLOBAL_NODE_COUNT;
  Inc(GLOBAL_NODE_COUNT);
end;

destructor TVariable.Cleanup;
var
  v: TVariable;
begin
  self.Data.Free;
  self.Grad.Free;
  self.TrackingID:='';
end;

procedure TVariable.AddPrev(AVariable: TVariable);
begin
  if not GLOBAL_SKIP_GRAD then
  begin
    SetLength(self.Prev, Length(self.Prev) + 1);
    self.Prev[Length(self.Prev) - 1] := AVariable;

    if AVariable.RequiresGrad then
      self.RequiresGrad:=True;
  end;
end;

procedure TVariable.AddPrev(arr: array of TVariable);
var
  T: TVariable;
begin
  for T in arr do
    self.AddPrev(T);
end;

procedure TVariable.Backpropagate;
begin
  BackwardGraph(self);
end;

procedure TVariable.FreeData;
begin
  self.Data.Free;
end;

procedure TVariable.FreeGrad;
begin
  self.Grad.Free;
end;

procedure TVariable.ZeroGrad;
var
  i: longint;
begin
  for i := 0 to self.Grad.Size - 1 do
    self.Grad.Val[i] := 0;
end;

function TVariable.Dot(Other: TVariable): TVariable;
begin
  Assert((Self.NDims <= 2) and (Other.NDims <= 2), MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := noe.Math.MatMul(self, Other);
end;

procedure ClearIntermediaryNodes;
var
  i: integer;
begin
  for i := 0 to length(GlobalNodeTracker.Items) - 1 do
    if not GlobalNodeTracker.Items[i].IsLeaf then
    begin
      GlobalNodeTracker.Items[i].FreeGrad;
      GlobalNodeTracker.Items[i].FreeData;
      GlobalNodeTracker.Items[i] := nil;
    end;
  SetLength(GlobalNodeTracker.Items, 0);
end;

procedure SetRequiresGrad(arr: array of TVariable; val: boolean);
var
  V: TVariable;
begin
  for V in arr do
    V.RequiresGrad := val;
end;

procedure ZeroGradGraph(const T: TVariable);
var
  arr: TVariableArr;
  i: integer;
begin
  arr := TopologicalSort(T);
  for i := 0 to length(arr) - 1 do
    arr[i].ZeroGrad;
end;

procedure Cleanup;
var
  N: TVariable;
begin
  for N in GlobalNodeTracker.NodeSpace do
    N.Cleanup;
end;

function CopyTensor(A: TTensor): TTensor;
begin
  Result.val := copy(A.val);
  Result.ReshapeInplace(A.Shape);
end;

function RandomTensorNormal(Shape: array of longint): TTensor;
var
  i: longint;
begin
  Result := CreateEmptyTensor(Shape);
  for i := 0 to Result.Size - 1 do
    Result.Val[i] := Math.randg(0, 1);
end;

function RandomTensorBinomial(Shape: array of longint; p: double): TTensor;
var
  i: longint;
begin
  Result := CreateEmptyTensor(Shape);
  for i := 0 to Result.Size - 1 do
    Result.Val[i] := ifthen(random > p, 0, 1);
end;

function ReadCSV(fileName: string; NRowSkip: longint): TTensor;
var
  s, number: string;
  sl: TStringList;
  InFile: Text;
  i, RowCount, ColCount, offset: longint;
begin
  Assert(FileExists(filename), 'File does not exist.');
  Assign(InFile, fileName);
  Reset(InFile);

  sl := TStringList.Create;
  sl.StrictDelimiter := True;

  { first run: estimate the RowCount & ColCount }
  ReadLn(InFile, s);
  sl.CommaText := s;
  ColCount     := sl.Count;

  RowCount := 1;
  while not EOF(InFile) do
  begin
    Inc(RowCount);
    ReadLn(InFile);
  end;

  Dec(RowCount, NRowSkip);

  { actual data handle }
  Result.ReshapeInplace([RowCount, ColCount]);
  SetLength(Result.Val, RowCount * ColCount);

  offset := 0;
  Reset(InFile);

  for i := 0 to NRowSkip - 1 do
    ReadLn(InFile);

  while not EOF(InFile) do
  begin
    ReadLn(InFile, s);
    sl.CommaText := s;

    for number in sl do
    begin
      Result.Val[offset] := StrToFloat(number);
      Inc(offset);
    end;
  end;

  Close(InFile);
  sl.Free;
end;

function CreateEmptyTensor(Shape: array of longint): TTensor;
begin
  Result := TTensor.Default;
  SetLength(Result.Val, ShapeToSize(Shape));
  Result.ReshapeInplace(shape);
  Result.Strides := ShapeToStride(Shape);
end;

function CreateTensor(Shape: array of longint; Val: NFloat): TTensor;
var
  i: longint;
begin
  Result := CreateEmptyTensor(Shape);
  for i := 0 to Result.Size - 1 do
    Result.Val[i] := Val;
end;

function CreateTensor(Shape: array of longint; Vals: array of NFloat): TTensor;
var
  i, size: longint;
begin
  size := ShapeToSize(Shape);
  Assert(ShapeToSize(Shape) = size,
    'The values cannot be reshaped into the target shape');
  Result := CreateEmptyTensor(shape);
  for i := 0 to size - 1 do
    Result.Val[i] := Vals[i];
  Result.ReshapeInplace(Shape);
end;

function Zeros(Shape: array of longint): TTensor;
begin
  Result := CreateTensor(Shape, 0);
end;

function Ones(Shape: array of longint): TTensor;
begin
  Result := CreateTensor(Shape, 1.0);
end;

function Range(start, stop, step: double): TTensor;
var
  i: double;
  offset: longint;
begin
  Result.ReshapeInplace([Ceil((stop - start) / step)]);
  Result.Strides := ShapeToStride([Ceil((stop - start) / step)]);
  SetLength(Result.Val, Ceil((stop - start) / step));

  i      := start;
  offset := 0;
  while offset < Ceil((stop - start) / step) do
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

function TopologicalSort(T: TVariable): TVariableArr;
var
  Seen, Sorted: TVariableArr;
  prv: TVariable;

  procedure TopoHelper(v: TVariable);
  begin
    if (not (v in Seen)) then
    begin
      SetLength(Seen, Length(seen) + 1);
      Seen[Length(Seen) - 1] := v;
      for prv in v.Prev do
        TopoHelper(prv);

      if v.RequiresGrad then
      begin
        SetLength(Sorted, Length(Sorted) + 1);
        Sorted[Length(Sorted) - 1] := v;
      end;
    end;
  end;

begin
  TopoHelper(T);
  Result := Sorted;
end;

procedure BackwardGraph(const T: TVariable);
var
  Sorted: TVariableArr;
  v: TVariable;
  i: longint;
begin
  if GLOBAL_SKIP_GRAD then
    exit;

  Sorted := TopologicalSort(T);

  T.Grad.ReshapeInplace(T.Data.Shape);
  T.Grad.Fill(1);

  for i := length(Sorted) - 1 downto 0 do
    if Assigned(Sorted[i].BackwardFunc) then
    begin
      Sorted[i].BackwardFunc(Sorted[i].Prev, Sorted[i].FGrad);
    end;

  GlobalNodeTracker.ClearUnusedNodes(T);
end;

function ShapeToStride(Shape: array of longint): TIntVector;
var
  k, j, sz, prod: longint;
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

function Squeeze(T: TTensor): TTensor;
var
  i, offset: longint;
  tmpShape: TIntVector;
begin
  Result := CopyTensor(T);
  SetLength(tmpShape, Length(T.Shape));

  offset := 0;
  for i in T.Shape do
    if i > 1 then
    begin
      tmpShape[offset] := i;
      Inc(offset);
    end;
  SetLength(tmpShape, offset);

  if Length(tmpShape) = 0 then
    Result.ReshapeInplace([1])
  else
    Result.ReshapeInplace(tmpShape);
end;

function GetRowRange(T: TTensor; RowIndex, Amount: longint): TTensor;
begin
  Assert(T.NDims = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := GetRange(T, RowIndex, 0, Amount, T.Shape[1]);
end;

function VFlip(T: TTensor): TTensor;
var
  i, j: longint;
begin
  Assert(T.NDims = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := CreateEmptyTensor(T.Shape);
  for i := 0 to T.Shape[0] - 1 do
    for j := 0 to T.Shape[1] - 1 do
      Result.SetAt(i, j, T.GetAt(T.Shape[0] - i - 1, j));
end;

function GetRange(T: TTensor; RowIndex, ColumnIndex, Height, Width: longint): TTensor;
var
  i, j, offset: longint;
begin
  Assert(Length(T.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result.ReshapeInplace([Height, Width]);

  SetLength(Result.Val, Height * Width);
  offset := 0;
  for i := RowIndex to RowIndex + Height - 1 do
    for j := ColumnIndex to ColumnIndex + Width - 1 do
    begin
      Result.Val[offset] := T.Val[i * T.Shape[1] + j];
      Inc(offset);
    end;
end;

function GetRange(T: TVariable;
  RowIndex, ColumnIndex, Height, Width: longint): TTensor;
begin
  Result := GetRange(T.Data, RowIndex, ColumnIndex, Height, Width);
end;

function GetColumn(T: TTensor; ColumnIndex: longint; KeepDims: boolean
  ): TTensor;
begin
  if not KeepDims then
    Exit(Squeeze(GetRange(T, 0, ColumnIndex, T.Shape[0], 1)))
  else
    Exit(GetRange(T, 0, ColumnIndex, T.Shape[0], 1));
end;

function GetColumnRange(T: TTensor; ColumnIndex, Amount: longint): TTensor;
begin
  Assert(T.NDims = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Result := GetRange(T, 0, ColumnIndex, T.Shape[0], Amount);
end;

function GetRow(T: TTensor; RowIndex: longint; KeepDims: boolean): TTensor;
begin
  if not KeepDims then
    Exit(Squeeze(GetRange(T, RowIndex, 0, 1, T.Shape[1])))
  else
    Exit(GetRange(T, RowIndex, 0, 1, T.Shape[1]));
end;

procedure PrintTensor(V: TVariable);
begin
  PrintTensor(V.Data);
end;

procedure IterateTensor(T, OutT: TTensor; Callback: TCallback);
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

          ithDimChanged := j; // in which dimension there is a change?
        end;

      //writeln(offset);
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
    currDim: longint; var T, OutT: TTensor);
begin
  OutT.Val[offset] := val;
end;
function AsStrided(X: TTensor; TargetShape, Strides: array of longint): TTensor;
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

function BroadcastTo(X: TTensor; TargetShape: array of longint): TTensor;
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

function IsBroadcastable(A, B: TTensor): boolean;
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

function TileColumn(A: TTensor; n: longint): TTensor;
var
  i, j: longint;
begin
  Result := CreateEmptyTensor([A.Shape[0], n]);
  for i := 0 to A.Shape[0] - 1 do
    for j := 0 to n-1 do
      result.Val[i * n + j] := A.val[i];
end;

function TileRow(A: TTensor; n: longint): TTensor;
var
  i, j, OutSize: longint;
begin
  OutSize := A.Size * n;
  Result := CreateEmptyTensor([n, A.Shape[1]]);
  i := 0;
  while i < OutSize do
  begin
    for j := 0 to A.Shape[1] - 1 do
      result.Val[i + j] := A.Val[j];
    i := i + A.Shape[1];
  end;
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

    ithDimChanged := n;
    for i := Length(res) - 1 downto 0 do
      if dimTracker[i] <> res[i] then
      begin
        dimTracker[i] := res[i];

        NewlineNum    := n - i - 1;
        ithDimChanged := i; // in which dimension there is a change?
      end;


    if (ithDimChanged < n - 1) then
      outstr := outstr + (DupeString(']', NewlineNum));

    outstr := outstr + (DupeString(sLineBreak, NewlineNum));

    if (ithDimChanged = n - 1) then
      outstr := outstr + (', ');

    if ithDimChanged < n - 1 then
    begin
      outstr := outstr + (DupeString(' ', n - NewlineNum));
      outstr := outstr + (DupeString('[', NewlineNum));
    end;

    outstr := outstr + Format('%'+IntToStr(digitMax+decimalPlace+2)+'.'+IntToStr(decimalPlace)+'f', [T.Val[IndexToOffset(res, T.Shape, T.Strides)]]);
  end;

  // d is dimension iterator, d=0..n-1
  procedure iterate(d: longint; shape, res: array of longint);
  var
    i: longint;
  begin
    if d >= n then
    begin
      //if (res[d-1] < 3) or (res[d-1] > T.Shape[n-1] - 3 - 1) then
        PPrint(res);

      //if res[d-1] = 3 then
      //  outstr := outstr + ', ... ';
      Inc(offset);
      exit;
    end;

    for i := 0 to shape[d] - 1 do
    begin
      res[d] := i;
      iterate(d + 1, shape, res);
    end;
  end;

  function MaxAbs(arr: array of NFloat): double;
  var
    i: double;
  begin
    Result := abs(arr[0]);
    for i in arr do
      if abs(i) > abs(Result) then
        Result := i;
  end;

begin
  digitMax     := Math.ceil(Math.log10(abs(MaxAbs(T.Val)) + 0.01));
  decimalPlace := 2;

  if Length(T.Val) = 1 then { it is a scalar }
    writeln(T.Val[0]: digitMax + decimalPlace + 1: decimalPlace)
  else { it is a higher rank tensor }
  begin
    offset := 0;
    n      := Length(T.Shape);

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
  NoeConfig.BLASFileName := BLAS_FILENAME;
  NoeConfig.useBLAS      := True;

  GlobalNodeTracker.NodeSpace := TVariableList.Create;

  GLOBAL_NODE_COUNT := 0;

finalization
  GlobalNodeTracker.NodeSpace.Free;
end.
