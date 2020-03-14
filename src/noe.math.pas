{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit extends FPC's math unit (and partially system unit) to be able to
 work with TTensors and TVariables.

 To do:
  - adapt more math functions from math.pas
  - implement iterate() that accepts callback to iterate over dimensions
  - provide backends for matrix transposition
  - Apply ReduceTo on the op that involves broadcasting
  - Fix broadcast guard :(( Shape[1000, 10] + [1, 11] passed :'(
}

unit noe.Math;

{$mode objfpc}{$H+}//{$INLINE ON}

interface

uses
  Classes, fgl, Math, noe, noe.backend.blas, noe.backend.native, noe.utils,
  RegExpr, strutils, SysUtils;

type
  { Wrapping FPC's f:R->R unary functions in math unit }
  TUFunc = function(v: NFloat): NFloat;

  { Wrapping FPC's f:RxR->R binary functions in math unit }
  TBFunc = function(v1, v2: NFloat): NFloat;

{ Helper to apply a function on each tensor's element }
function ApplyUfunc(A: TTensor; Func: TUFunc): TTensor;
function ApplyBfunc(A, B: TTensor; Func: TBFunc): TTensor;
function IsBlasfuncAvailable(Func: Pointer): boolean;

{ Some of functions belong to system unit are in different format. Hence, they
  need to be wrapped to make them compatible. They are given suffix "F"
  (indicating NFloat-valued function) to avoid confusion. }
function Sin_F(x: NFloat): NFloat;
function Cos_F(x: NFloat): NFloat;
function Cosh_F(x: NFloat): NFloat;
function Exp_F(x: NFloat): NFloat;
function Ln_F(x: NFloat): NFloat;
function Log2_F(x: NFloat): NFloat;
function Log10_F(x: NFloat): NFloat;
function Add_F(v1, v2: NFloat): NFloat;
function Subtract_F(v1, v2: NFloat): NFloat;
function Divide_F(v1, v2: NFloat): NFloat;
function DegToRad_F(x: NFloat): NFloat;
function Multiply_F(v1, v2: NFloat): NFloat;
function Power_F(v1, v2: NFloat): NFloat;
function RadToDeg_F(x: NFloat): NFloat;
function Sinh_F(x: NFloat): NFloat;
function Tan_F(x: NFloat): NFloat;
function Tanh_F(x: NFloat): NFloat;

{ TTensor math ----------------------------------------------------------------}

function Add(A, B: TTensor): TTensor;
function ArgMax(M: TTensor): TTensor;
function ArgMax(M: TTensor; axis: byte): TTensor; overload;
function Conv2D(X, w: TTensor;
  PaddingHeight, PaddingWidth, StrideHeight, StrideWidth: longint): TTensor;
function Cos(A: TTensor): TTensor;
function Cosh(A: TTensor): TTensor;
function DegToRad(A: TTensor): TTensor;
function Divide(A, B: TTensor): TTensor;
function Exp(A: TTensor): TTensor;
function LeakyReLU(A: TTensor; v: NFloat): TTensor;
function Log10(A: TTensor): TTensor;
function Log2(A: TTensor): TTensor;
function Log(A: TTensor): TTensor;
function MatMul(A, B: TTensor): TTensor;
function Min(M: TTensor): TTensor;
function Max(M: TTensor): TTensor;
function Max(M: TTensor; axis: byte): TTensor;
function Mean(M: TTensor): TTensor;
function Mean(M: TTensor; axis: byte): TTensor;
function Multiply(A, B: TTensor): TTensor;
function Power(A: TTensor; exponent: NFloat): TTensor; overload;
function Power(A, B: TTensor): TTensor; overload;
function RadToDeg(A: TTensor): TTensor;
function ReLU(T: TTensor): TTensor;
function Sin(A: TTensor): TTensor;
function Sigmoid(A: TTensor): TTensor;
function Sinh(A: TTensor): TTensor;
function SoftMax(A: TTensor; axis: byte): TTensor;
function Subtract(A, B: TTensor): TTensor;
function Sum(M: TTensor): TTensor;
function Sum(M: TTensor; axis: byte): TTensor; overload;
function Sum(X: TTensor; dim: longint; KeepDims: boolean = False): TTensor;
function Sum(X: TTensor; dims: array of longint; KeepDims: boolean = False): TTensor;
function Tan(A: TTensor): TTensor;
function Tanh(A: TTensor): TTensor;
function Transpose(T: TTensor; dims: array of longint): TTensor;
function Transpose(T: TTensor): TTensor;
function Transpose2D(T: TTensor): TTensor;
function TransposeTensor(X: TTensor; axis: array of longint): TTensor;


{ Evaluates the Einstein summation convention on the operands. Very slow now.
  The initial implementation is heavily inspired from Kyle Hundman's attempt to
  mirror numpy's einsum, so not all operations are supported. Any helps are
  welcome. }
function Einsum(Subscripts: string; Pots: array of TTensor): TTensor;

 { TVariable math --------------------------------------------------------------}
 { forward mode }
function Add(A, B: TVariable): TVariable;
function Conv2D(X, w: TVariable;
  PaddingHeight, PaddingWidth, StrideHeight, StrideWidth: longint): TVariable;
function Cosh(A: TVariable): TVariable;
function Divide(A, B: TVariable): TVariable;
function Exp(A: TVariable): TVariable;
function LeakyReLU(A: TVariable; v: NFloat): TVariable; overload;
function Log(A: TVariable): TVariable;
function Max(A: TVariable): TVariable;
function Max(A: TVariable; axis: byte): TVariable; overload;
function Mean(A: TVariable; axis: byte): TVariable;
function Mean(A: TVariable): TVariable; overload;
function Multiply(A, B: TVariable): TVariable;
function MultiplyC(A: TVariable; x: NFloat): TVariable;
function MatMul(A, B: TVariable): TVariable;
function Negate(A: TVariable): TVariable;
function ReLU(A: TVariable): TVariable;
function Reshape(A: TVariable; Shape: array of longint): TVariable;
function Sigmoid(A: TVariable): TVariable;
function Sinh(A: TVariable): TVariable;
function Sqr(A: TVariable): TVariable;
function Sqrt(A: TVariable): TVariable;
function Subtract(A, B: TVariable): TVariable;
function Sum(A: TVariable; axis: byte): TVariable;
function Sum(A: TVariable): TVariable; overload;
function Tanh(A: TVariable): TVariable;

{ backward mode }
procedure BackwardAdd(arr: TVariableArr; ADy: TTensor);
procedure BackwardConv2D(arr: TVariableArr; ADy: TTensor);
procedure BackwardDivide(arr: TVariableArr; ADy: TTensor);
procedure BackwardSubtract(arr: TVariableArr; ADy: TTensor);
procedure BackwardMultiply(arr: TVariableArr; ADy: TTensor);
procedure BackwardMultiplyC(arr: TVariableArr; ADy: TTensor);
procedure BackwardMatmul(arr: TVariableArr; ADy: TTensor);
procedure BackwardCosh(arr: TVariableArr; ADy: TTensor);
procedure BackwardLeakyReLU(arr: TVariableArr; ADy: TTensor);
procedure BackwardLn(arr: TVariableArr; ADy: TTensor);
procedure BackwardExp(arr: TVariableArr; ADy: TTensor);
procedure BackwardMax(arr: TVariableArr; ADy: TTensor);
procedure BackwardMean(arr: TVariableArr; ADy: TTensor);
procedure BackwardNegate(arr: TVariableArr; ADy: TTensor);
procedure BackwardReLU(arr: TVariableArr; ADy: TTensor);
procedure BackwardReshape(arr: TVariableArr; ADy: TTensor);
procedure BackwardSigmoid(arr: TVariableArr; ADy: TTensor);
procedure BackwardSinh(arr: TVariableArr; ADy: TTensor);
procedure BackwardSqr(arr: TVariableArr; ADy: TTensor);
procedure BackwardSqrt(arr: TVariableArr; ADy: TTensor);
procedure BackwardSum(arr: TVariableArr; ADy: TTensor);
procedure BackwardTanh(arr: TVariableArr; ADy: TTensor);

{ aggregate functions, derived from above functions }
function SoftMax(A: TVariable; axis: byte): TVariable;

{ If target is the result of broadcasting, reduce to its original shape }
function ReduceTo(Target, Other: TTensor): TTensor;

procedure CopyArrayAt(var Src, Dest: TFloatVector; offset: longint);

function Col2Im(imgcol: TTensor;
  Channels, Height, Width, FilterH, FilterW, PaddingHeight, PaddingWidth,
  StrideHeight, StrideWidth: longint): TTensor;

function Col2ImBatch(imgcol: TTensor;
  Channels, Height, Width, FilterH, FilterW, PaddingHeight, PaddingWidth,
  StrideHeight, StrideWidth: longint): TTensor;

function Im2Col(img: TTensor;
  Channels, Height, Width, FilterH, FilterW, PaddingHeight, PaddingWidth,
  StrideHeight, StrideWidth: longint): TTensor;

function Im2ColBatch(X: TTensor;
  FilterH, FilterW, PaddingHeight, PaddingWidth, StrideHeight, StrideWidth:
  longint): TTensor;


implementation

function Log2_F(x: NFloat): NFloat;
begin
  Result := Math.log2(x);
end;

function Log10_F(x: NFloat): NFloat;
begin
  Result := Math.log10(x);
end;

function Add_F(v1, v2: NFloat): NFloat;
begin
  Result := v1 + v2;
end;

function Subtract_F(v1, v2: NFloat): NFloat;
begin
  Result := v1 - v2;
end;

function Divide_F(v1, v2: NFloat): NFloat;
begin
  Result := v1 / v2;
end;

function DegToRad_F(x: NFloat): NFloat;
begin
  Result := Math.degtograd(x);
end;

function Multiply_F(v1, v2: NFloat): NFloat;
begin
  Result := v1 * v2;
end;

function Power_F(v1, v2: NFloat): NFloat;
begin
  Result := Math.power(v1, v2);
end;

function RadToDeg_F(x: NFloat): NFloat;
begin
  Result := Math.radtodeg(x);
end;

function Sinh_F(x: NFloat): NFloat;
begin
  Result := Math.sinh(x);
end;

function Tan_F(x: NFloat): NFloat;
begin
  Result := Math.tan(x);
end;

function Tanh_F(x: NFloat): NFloat;
begin
  Result := Math.tanh(x);
end;

function Add(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @Add_F);
end;

//function Convolve2D(A, w: TTensor): TTensor;
//begin
//  raise ENotImplemented.Create('Not implemented');
//end;

function Subtract(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @Subtract_F);
end;

function Divide(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @Divide_F);
end;

function Multiply(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @Multiply_F);
end;

function MatMul(A, B: TTensor): TTensor;
begin
  Assert((length(A.Shape) <= 2) and (length(B.Shape) <= 2),
    'Tensor dimension must be <= 2.');

  { calculates matrix multiplication according to the backend }
  if IsBlasfuncAvailable(blas_dgemm) then
    Result := MatMul_BLAS(A, B)
  else
    Result := MatMul_Native(A, B);
end;

function ArgMax(M: TTensor): TTensor;
begin
  SetLength(Result.Val, 1);
  Result.Val[0] := ArgMax(M.Val);
  Result.ReshapeInplace([1]);
end;

function ArgMax(M: TTensor; axis: byte): TTensor;
var
  i: integer;
begin
  Assert(Length(M.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Assert(axis in [0, 1], MSG_ASSERTION_INVALID_AXIS);
  if axis = 0 then
  begin
    SetLength(Result.Val, M.Shape[1]);
    Result.ReshapeInplace([1, M.Shape[1]]);
    for i := 0 to M.Shape[1] - 1 do
      Result.Val[i] := ArgMax(GetColumn(M, i).Val);
  end
  else
  begin
    SetLength(Result.Val, M.Shape[0]);
    Result.ReshapeInplace([M.Shape[0], 1]);
    for i := 0 to M.Shape[0] - 1 do
      Result.Val[i] := ArgMax(GetRow(M, i).Val);
  end;
end;

function Min(M: TTensor): TTensor;
begin
  SetLength(Result.Val, 1);
  Result.Val[0] := MinValue(M.Val);
  Result.ReshapeInplace([1]);
end;

function Max(M: TTensor): TTensor;
begin
  SetLength(Result.Val, 1);
  Result.Val[0] := MaxValue(M.Val);
  Result.ReshapeInplace([1]);
end;

function Max(M: TTensor; axis: byte): TTensor;
var
  i: longint;
begin
  Assert(Length(M.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Assert(axis in [0, 1], MSG_ASSERTION_INVALID_AXIS);
  if axis = 0 then
  begin
    SetLength(Result.Val, M.Shape[1]);
    Result.ReshapeInplace([1, M.Shape[1]]);
    for i := 0 to M.Shape[1] - 1 do
      Result.Val[i] := MaxValue(GetColumn(M, i).Val);
  end
  else
  begin
    SetLength(Result.Val, M.Shape[0]);
    Result.ReshapeInplace([M.Shape[0], 1]);
    for i := 0 to M.Shape[0] - 1 do
      Result.Val[i] := MaxValue(GetRow(M, i).Val);
  end;
end;

function Mean(M: TTensor): TTensor;
var
  i: longint;
  tot: single;
begin
  tot := 0;
  for i := 0 to length(M.Val) - 1 do
    tot  := tot + M.val[i];
  Result := tot / Length(M.Val);
end;

function Mean(M: TTensor; axis: byte): TTensor;
begin
  Assert(axis <= 1, MSG_ASSERTION_INVALID_AXIS);
  if axis = 0 then
  begin
    if IsBlasfuncAvailable(blas_dgemm) then
      Result := MeanCol_BLAS(M)
    else
      Result := MeanCol_Native(M);
  end
  else
  if IsBlasfuncAvailable(blas_dgemm) then
    Result := MeanRow_BLAS(M)
  else
    Result := MeanRow_Native(M);
end;

function SoftMax(A: TTensor; axis: byte): TTensor;
var
  X, Y: TTensor;
begin
  X      := A - Max(A, axis);
  Y      := Exp(X);
  Result := Y / sum(Y, axis);
end;

function Sum(M: TTensor): TTensor;
var
  i: longint;
  tot: single;
begin
  tot := 0;
  for i := 0 to length(M.Val) - 1 do
    tot  := tot + M.val[i];
  Result := tot;
end;

function Sum(M: TTensor; axis: byte): TTensor;
begin
  Assert(axis <= 1, MSG_ASSERTION_INVALID_AXIS);
  if axis = 0 then
  begin
    if IsBlasfuncAvailable(blas_daxpy) then
      Result := SumCol_BLAS(M)
    else
      Result := SumCol_Native(M);
  end
  else
  if IsBlasfuncAvailable(blas_daxpy) then
    Result := SumRow_BLAS(M)
  else
    Result := SumRow_Native(M);
end;

function DegToRad(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @DegToRad_F);
end;

function RadToDeg(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @RadToDeg_F);
end;

function LeakyReLU(A: TTensor; v: NFloat): TTensor;
var
  i: longint;
begin
  Result := CreateEmptyTensor(A.Shape);
  for i := 0 to A.Size - 1 do
    Result.Val[i] := IfThen(A.Val[i] < 0, A.Val[i] * v, A.Val[i]);
end;

function Log10(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Log10_F);
end;

function Log2(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Log2_F);
end;

function Log(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Ln_F);
end;

function Transpose(T: TTensor; dims: array of longint): TTensor;
begin
  Assert(Length(dims) = length(T.Shape),
    'dims length does not match tensor dimension');
  Result := TransposeTensor(T, dims);
end;

function Transpose(T: TTensor): TTensor;
var
  OutDims: TIntVector;
  i: longint;
begin
  // attempt with 2d transpose first
  if (Length(T.Shape) = 2) then
    Result := Transpose2D(T)
  else
  begin
    SetLength(OutDims, T.NDims);
    for i := 0 to T.NDims - 1 do
      OutDims[i] := T.NDims - i - 1;
    Result := TransposeTensor(T, OutDims);
  end;
end;

function Transpose2D(T: TTensor): TTensor;
var
  i, j: longint;
begin
  Assert(Length(T.Shape) = 2, 'Transpose2D only accepts rank-2 tensors');
  Result.ReshapeInplace([T.Shape[1], T.Shape[0]]);
  SetLength(Result.Val, Length(T.Val));
  for i := 0 to T.Shape[0] - 1 do
    for j := 0 to T.Shape[1] - 1 do
      Result.Val[j * T.Shape[0] + i] := T.Val[i * T.Shape[1] + j];
end;

procedure cbTranspose(val: NFloat; offset: longint; idx: TIntVector;
  currDim: longint; var T, OutT: TTensor);
begin
  OutT.Val[offset] := val;
end;

function TransposeTensor(X: TTensor; axis: array of longint): TTensor;
var
  outStrides, OutShape: TIntVector;
  i: integer;
begin
  SetLength(OutShape, Length(axis));
  SetLength(OutStrides, Length(axis));
  SetLength(Result.Val, X.Size);

  for i := 0 to Length(axis) - 1 do
  begin
    OutShape[i]   := X.Shape[axis[i]];
    OutStrides[i] := X.Strides[axis[i]];
  end;

  X.ReshapeInplace(OutShape);
  X.Strides := outStrides;

  { Fill Result.Val with strided X.Val }
  IterateTensor(X, Result, @cbTranspose);

  Result.ReshapeInplace(OutShape);
  //Result.Strides := outStrides;
end;

function ReLU(T: TTensor): TTensor;
var
  i: longint;
begin
  Result.ReshapeInplace(T.Shape);
  SetLength(Result.Val, Length(T.Val));
  for i := 0 to Length(Result.Val) - 1 do
    Result.Val[i] := Max(0, T.Val[i]);
end;

function Sin_F(x: NFloat): NFloat;
begin
  Result := System.Sin(x);
end;

function Cos_F(x: NFloat): NFloat;
begin
  Result := System.Cos(x);
end;

function Cosh_F(x: NFloat): NFloat;
begin
  Result := Math.Cosh(x);
end;

function Exp_F(x: NFloat): NFloat;
begin
  Result := System.exp(x);
end;

function Ln_F(x: NFloat): NFloat;
begin
  Result := system.ln(Math.Max(x, EPS_TOL));
end;

function Sin(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Sin_F);
end;

function Sigmoid(A: TTensor): TTensor;
begin
  Result := 1 / (1 + Exp(-A));
end;

function Sinh(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Sinh_F);
end;

{ copy entire items in `Src` to `Dest` at position `offset` }
procedure CopyArrayAt(var Src, Dest: TFloatVector; offset: longint);
var
  i: longint;
begin
  for i := 0 to High(Src) do
    Dest[i + offset] := Src[i];
end;

function Conv2D(X, w: TTensor;
  PaddingHeight, PaddingWidth, StrideHeight, StrideWidth: longint): TTensor;
var
  m, i, Channels, Height, Width, ConvOutHeight, ConvOutWidth, FilterH,
  FilterW, offset: longint;
  W_, res: TTensor;
begin
  m      := X.Shape[0];
  Channels := X.Shape[1];
  Height := X.Shape[2];
  Width  := X.Shape[3];
  FilterH := W.Shape[2];
  FilterW := W.Shape[3];
  ConvOutHeight := (Height + 2 * PaddingHeight - FilterH) div StrideHeight + 1;
  ConvOutWidth := (Width + 2 * PaddingWidth - FilterW) div StrideWidth + 1;
  W_     := W.Reshape([W.Shape[0], ConvOutHeight * ConvOutWidth]);

  SetLength(Result.Val, m * W.Shape[0] * ConvOutHeight *
    ConvOutWidth);

  offset := 0;
  for i := 0 to m - 1 do
  begin
    res := W_.Dot(Im2Col(X.GetAt([i]), Channels, Height, Width, FilterH,
      FilterW, PaddingHeight, PaddingWidth, StrideHeight, StrideWidth)).T;

    CopyArrayAt(res.Val, Result.Val, offset);

    offset := offset + res.Size;
  end;

  {plus}
  {bias}
  {here}

  Result.ReshapeInplace([m, res.Shape[0], res.Shape[1]]);
  Result := Transpose(Result, [0, 2, 1]);
  Result.ReshapeInplace([m, W.Shape[0], ConvOutWidth, ConvOutHeight]);
end;

function Cos(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Cos_F);
end;

function Cosh(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Cosh_F);
end;

function Sum(X: TTensor; dim: longint; KeepDims: boolean): TTensor;
var
  i, j, tmp, DimSize, remSize, outSize, Offset, rep, stride: longint;
  outShape, newDims: TIntVector;
  tmpTensor: TTensor;
begin
  Assert(dim < X.NDims, 'Dimension out of bound');

  { HACK: if `dim` is the highest dimension, permute tensor w.r.t. two last
    dimensions. }
  if dim = X.NDims - 1 then
  begin
    dim := dim - 1;
    SetLength(newDims, X.NDims);
    for i := 0 to X.NDims - 1 do
      newDims[i] := i;
    tmp := newDims[High(newDims)];
    newDims[High(newDims)] := newDims[High(newDims) - 1];
    newDims[High(newDims) - 1] := tmp;
    X   := Transpose(X, newDims);
  end;

  { The first `dim` is a special case }
  if dim > 0 then
    stride := X.Strides[dim - 1]
  else
    stride := X.Strides[dim];

  { number of 'raw' rows }
  rep := 1;
  if (X.NDims > 1) and (dim <> 0) then
    if (dim = high(X.Shape)) then
      rep := X.Shape[dim - 1]
    else if (dim = 0) then
      rep := X.Shape[dim + 1]
    else
      for i := 0 to dim - 1 do
        rep := rep * X.Shape[i];

  { given the offset at which we start, determine the block size that should
    be grouped together }
  remSize := 1;
  for i := dim + 1 to X.NDims - 1 do
    remSize := remSize * X.Shape[i];

  outSize  := 1;    // the resulting size (of `Result.Val`)
  outShape := nil;  // the resulting shape
  for i := 0 to High(X.Shape) do
    if dim <> i then
    begin
      outSize := outSize * X.Shape[i];
      SetLength(outShape, Length(outShape) + 1);
      outShape[Length(outShape) - 1] := X.Shape[i];
    end
    { do not squeeze w.r.t. `dim` if `KeepDims` is true. }
    else if KeepDims then
    begin
      SetLength(outShape, Length(outShape) + 1);
      outShape[Length(outShape) - 1] := 1;
    end;

  SetLength(Result.Val, outSize);
  DimSize := X.Shape[dim];
  Offset  := 0;
  for j := 0 to rep - 1 do
  begin
    tmpTensor := Zeros([remSize]);
    for i := 0 to DimSize - 1 do
    begin
      tmpTensor := tmpTensor + CreateTensor([remSize], Copy(X.Val, i * X.Strides[dim] +
        j * stride, remSize));
      CopyArrayAt(tmpTensor.Val, Result.Val, Offset);
    end;
    Inc(Offset, remSize);
  end;

  Result.ReshapeInplace(outShape);
end;

function Sum(X: TTensor; dims: array of longint; KeepDims: boolean): TTensor;
var
  i, dim: integer;
  outShape: TIntVector;
begin
  Result := X;
  for dim in dims do
    Result := Sum(Result, dim, True);

  outShape := nil;

  if KeepDims then
    SetLength(outShape, X.NDims);

  for i := 0 to X.NDims - 1 do
    if not KeepDims then
    begin
      if not (i in dims) then
      begin
        SetLength(outShape, Length(outShape) + 1);
        outShape[Length(outShape) - 1] := X.Shape[i];
      end;
    end
    else
    if (i in dims) then
      outShape[i] := 1
    else
      outShape[i] := X.Shape[i];

  Result.ReshapeInplace(outShape);
end;

function Tan(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Tan_F);
end;

function Tanh(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Tanh_F);
end;

function Power(A: TTensor; exponent: NFloat): TTensor;
begin
  Result := ApplyBfunc(A, exponent, @Power_F);
end;

function Power(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @Power_F);
end;

function Exp(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Exp_F);
end;

procedure CreateOrUpdateOpNode(var res: TVariable; ResultName: string;
  inputs: array of TVariable; Data: TTensor; BackwardFunc: TBackwardFunc);
var
  TrackingID: string;
  i, TrackedNodeIdx: longint;
begin
  TrackingID := ResultName + ';';
  for i := 0 to Length(inputs) - 1 do
  begin
    TrackingID := TrackingID + IntToStr(inputs[i].ID);
    if i < Length(inputs) - 1 then
      TrackingID := TrackingID + ';';
  end;
  TrackedNodeIdx := GlobalNodeTracker.FindByTrackingID(TrackingID);

  if TrackedNodeIdx > -1 then
  begin
    Res      := GlobalNodeTracker.Items[TrackedNodeIdx];
    Res.Data := Data;
  end
  else
  begin
    Res := TVariable.Create(Data, ResultName, BackwardFunc);
    Res.TrackingID := TrackingID;
    Res.AddPrev(inputs);

    { track the non-leaf nodes }
    GlobalNodeTracker.Add(Res);
  end;
end;

function Add(A, B: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardAdd', [A, B], (A.Data + B.Data), @BackwardAdd);
end;

function Divide(A, B: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardDivide', [A, B], (A.Data / B.Data),
    @BackwardDivide);
end;

function Subtract(A, B: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSubtract', [A, B], (A.Data - B.Data),
    @BackwardSubtract);
end;

function Multiply(A, B: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardMultiply', [A, B], (A.Data * B.Data),
    @BackwardMultiply);
end;

function MultiplyC(A: TVariable; x: NFloat): TVariable;
begin
  Result := TVariable.Create(noe.Math.Multiply(A.Data, x), 'ForwardMultiplyC',
    @BackwardMultiplyC);

  SetLength(Result.Prev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := TVariable.Create(x, '');
  Result.Prev[1].RequiresGrad := False;
end;

function MatMul(A, B: TVariable): TVariable;
begin
  Assert(A.Shape[1] = B.Shape[0], MSG_ASSERTION_DIM_MISMATCH);

  CreateOrUpdateOpNode(Result, 'ForwardMatMul', [A, B], MatMul(A.Data, B.Data),
    @BackwardMatmul);

end;

function Negate(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardNegate', [A], -A.Data, @BackwardNegate);
end;

function Conv2D(X, w: TVariable;
  PaddingHeight, PaddingWidth, StrideHeight, StrideWidth: longint): TVariable;
var
  HOut, WOut: longint;
  XCol, WCol, XOut: TTensor;
begin
  HOut := (X.Shape[2] - w.Shape[2] + 2 * PaddingHeight) div StrideHeight + 1;
  WOut := (X.Shape[3] - w.Shape[3] + 2 * PaddingWidth) div StrideWidth + 1;

  XCol := Im2ColBatch(X.Data, w.Shape[2], w.Shape[3], PaddingHeight,
    PaddingWidth, StrideHeight, StrideWidth);
  WCol := w.Data.Reshape([w.Shape[0], w.Shape[1] * w.Shape[2] * w.Shape[3]]);
  XOut := WCol.Dot(XCol).Reshape([w.Shape[0], HOut, WOut, X.Shape[0]]);
  XOut := Transpose(XOut, [3, 0, 1, 2]);

  //writeln('im2col shape: ', XCol.Shape[0], ' ', XCol.Shape[1]);

  CreateOrUpdateOpNode(Result, 'ForwardConv2D', [X, w, PaddingHeight,
    PaddingWidth, StrideHeight, StrideWidth, XCol],
    XOut, @BackwardConv2D);
end;

function Cosh(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardCosh', [A], Cosh(A.Data), @BackwardCosh);
end;

function LeakyReLU(A: TVariable; v: NFloat): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardLeakyReLU', [A, v], LeakyReLU(A.Data, v),
    @BackwardLeakyReLU);
end;

function Log(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardLog', [A], Log(A.Data), @BackwardLn);
end;

function Reshape(A: TVariable; Shape: array of longint): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardReshape', [A], A.Data.Reshape(Shape), @BackwardReshape);
end;

function Sigmoid(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSigmoid', [A], Sigmoid(A.Data), @BackwardSigmoid);
end;

function Sinh(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSinh', [A], Sinh(A.Data), @BackwardSinh);
end;

function Sqr(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSqr', [A], (A.Data ** 2), @BackwardSqr);
end;

function Sqrt(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSqrt', [A], (A.Data ** 0.5), @BackwardSqrt);
end;

function ReLU(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardReLU', [A], ReLU(A.Data), @BackwardReLU);
end;

function Tanh(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardTanh', [A], Tanh(A.Data), @BackwardTanh);
end;

function Exp(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardExp', [A], Exp(A.Data), @BackwardExp);
end;

function Max(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardMax', [A], Max(A.Data), @BackwardMax);
end;

function Max(A: TVariable; axis: byte): TVariable;
begin
  { Max along axis has no gradient (?) }
  Result := TVariable.Create(Max(A.Data, axis), 'ForwardMax');
end;

function Mean(A: TVariable; axis: byte): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardMean', [A, axis], Mean(A.Data, axis),
    @BackwardMean);
end;

function Mean(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardMean', [A], Mean(A.Data), @BackwardMean);
end;

function Sum(A: TVariable; axis: byte): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSum', [A, axis], Sum(A.Data, axis), @BackwardSum);
end;

function Sum(A: TVariable): TVariable;
begin
  CreateOrUpdateOpNode(Result, 'ForwardSum', [A], Sum(A.Data), @BackwardSum);
end;

function SoftMax(A: TVariable; axis: byte): TVariable;
var
  X, Y: TVariable;
begin
  Result := Exp((A - Max(A, axis))) / sum(Exp((A - Max(A, axis))), axis);
end;

function ReduceTo(Target, Other: TTensor): TTensor;
var
  i: integer;
  dims, shape1, shape2: array of longint;
begin
  Result := Target;

  shape1 := (Target.Shape);
  shape2 := (Other.Shape);

  SetLength(shape1, Math.Max(Length(shape1), length(shape2)));
  SetLength(shape2, Math.Max(Length(shape1), length(shape2)));

  dims := nil;
  for i := 0 to Length(shape2) - 1 do
    if Shape2[i] <> Shape1[i] then
    begin
      SetLength(dims, Length(dims) + 1);
      dims[Length(dims) - 1] := i;
    end;

  if Length(dims) > 0 then
  begin
    Result := Sum(Result, dims);
    Result.ReshapeInplace(other.shape);
  end;
end;

function Im2ColGetPixel(img: TTensor;
  Height, Width, channels, row, col, channel, padH, padW: longint): NFloat;
var
  r, c: longint;
begin
  r := row - padH;
  c := col - padW;

  if ((r < 0) or (c < 0) or (r >= Height) or (c >= Width)) then
    Exit(0);
  Exit(img.Val[c + Width * (r + Height * channel)]);
end;

function Im2ColGetPixel(img: TTensor;
  imgIdx, Height, Width, channels, row, col, channel, padH, padW: longint): NFloat;
var
  r, c: longint;
begin
  r := row - padH;
  c := col - padW;

  writeln(Height);

  if ((r < 0) or (c < 0) or (r >= Height) or (c >= Width)) then
    Exit(0);
  Exit(img.Val[c + Width * (r + Height * channel)]);
end;

procedure Col2ImAddPixel(var img: TTensor;
  Height, Width, channels, row, col, channel, padH, padW: longint; val: NFloat);
var
  r, c: longint;
begin
  r := row - padH;
  c := col - padW;

  if ((r < 0) or (c < 0) or (r >= Height) or (c >= Width)) then
    Exit;
  img.Val[c + Width * (r + Height * channel)] := val;
end;

function Col2Im(imgcol: TTensor;
  Channels, Height, Width, FilterH, FilterW, PaddingHeight, PaddingWidth,
  StrideHeight, StrideWidth: longint): TTensor;
var
  ColHeight, ColWidth: longint;
  ChannelsCol, c, h, w, wOffset, hOffset, cIm, hIm, wIm: longint;
begin
  ColHeight   := (Height + 2 * PaddingHeight - FilterH) div StrideHeight + 1;
  ColWidth    := (Width + 2 * PaddingWidth - FilterW) div StrideWidth + 1;
  ChannelsCol := Channels * FilterH * FilterW;

  SetLength(Result.Val, Channels * Height * Width);
  Result.ReshapeInplace([Channels, Height, Width]);

  for c := 0 to ChannelsCol - 1 do
  begin
    wOffset := c mod FilterW;
    hOffset := (c div FilterW) mod FilterH;
    cIm     := c div FilterH div FilterW;
    for h := 0 to ColHeight - 1 do
    begin
      hIm := h * StrideHeight - PaddingHeight + hOffset;
      for w := 0 to ColWidth - 1 do
      begin
        wIm := w * StrideWidth - PaddingWidth + wOffset;

        if (hIm >= 0) and (hIm < Height) and (wIm >= 0) and (wIm < Width) then
          Result.Val[(cIm * Height + hIm) * Width + wIm] :=
            Result.Val[(cIm * Height + hIm) * Width + wIm] + imgcol.Val[
            (c * ColHeight + h) * ColWidth + w];
      end;
    end;
  end;

end;

function Col2ImBatch(imgcol: TTensor;
  Channels, Height, Width, FilterH, FilterW, PaddingHeight, PaddingWidth,
  StrideHeight, StrideWidth: longint): TTensor;
var
  ConvOutHeight, ConvOutWidth, sz, i, m, Offset: longint;
  tmpImgCol: TTensor;
begin
  ConvOutHeight := (Height + 2 * PaddingHeight - FilterH) div StrideHeight + 1;
  ConvOutWidth  := (Width + 2 * PaddingWidth - FilterW) div StrideWidth + 1;

  { size of a single im2col }
  sz := Channels * FilterH * FilterW * ConvOutHeight * ConvOutWidth;

  { number of sample }
  m := imgcol.Size div sz;

  tmpImgCol := Transpose(imgcol.Reshape([Channels * FilterH * FilterW,
    m, ConvOutHeight * ConvOutWidth]), [1, 0, 2]);

  Offset := 0;
  SetLength(Result.Val, m * Channels * Height * Width);
  for i := 0 to m - 1 do
  begin
    CopyArrayAt(Col2Im(tmpImgCol.GetAt([i]), Channels, Height, Width,
      FilterH, FilterW, PaddingHeight, PaddingWidth, StrideHeight, StrideWidth).val,
      Result.Val, Offset);
    Inc(Offset, Channels * Height * Width);
  end;
  Result.ReshapeInplace([m, Channels, Height, Width]);
end;

function Im2Col(img: TTensor;
  Channels, Height, Width, FilterH, FilterW, PaddingHeight, PaddingWidth,
  StrideHeight, StrideWidth: longint): TTensor;
var
  ConvOutHeight, ConvOutWidth: longint;
  ChannelsCol, c, h, w, wOffset, hOffset, cIm: longint;
  ImRow, ImCol, colIdx: longint;
begin
  ConvOutHeight := (Height + 2 * PaddingHeight - FilterH) div StrideHeight + 1;
  ConvOutWidth  := (Width + 2 * PaddingWidth - FilterW) div StrideWidth + 1;
  ChannelsCol   := Channels * FilterH * FilterW;

  SetLength(Result.Val, Channels * FilterH * FilterW * ConvOutHeight * ConvOutWidth);
  Result.ReshapeInplace([Channels * FilterH * FilterW, ConvOutHeight * ConvOutWidth]);
  for c := 0 to ChannelsCol - 1 do
  begin
    wOffset := c mod FilterW;
    hOffset := (c div FilterW) mod FilterH;
    cIm     := c div FilterH div FilterW;
    for h := 0 to ConvOutHeight - 1 do
      for w := 0 to ConvOutWidth - 1 do
      begin
        ImRow  := hOffset + h * StrideHeight;
        ImCol  := wOffset + w * StrideWidth;
        colIdx := (c * ConvOutHeight + h) * ConvOutWidth + w;

        Result.Val[colIdx] := Im2ColGetPixel(img, Height, Width,
          Channels, ImRow, ImCol, cIm, PaddingHeight, PaddingWidth);
      end;
  end;
end;

function Im2ColBatch(X: TTensor;
  FilterH, FilterW, PaddingHeight, PaddingWidth, StrideHeight, StrideWidth:
  longint): TTensor;
var
  m, ConvOutHeight, ConvOutWidth: longint;
  i: longint;
  Height, Width, channels, Offset, sz: longint;
begin
  m      := X.Shape[0];
  Height := X.Shape[2];
  Width  := X.Shape[3];
  channels := X.Shape[1];
  ConvOutHeight := (Height + 2 * PaddingHeight - FilterH) div StrideHeight + 1;
  ConvOutWidth := (Width + 2 * PaddingWidth - FilterW) div StrideWidth + 1;

  { size of a single im2col }
  sz := Channels * FilterH * FilterW * ConvOutHeight * ConvOutWidth;

  SetLength(Result.Val, sz * m);
  Result.ReshapeInplace([m, Channels * FilterH * FilterW, ConvOutHeight * ConvOutWidth]);
  Offset := 0;
  for i := 0 to m - 1 do
  begin
    CopyArrayAt(Im2Col(X.GetAt([i]), channels, Height, Width, FilterH,
      FilterW, PaddingHeight, PaddingWidth, StrideHeight, StrideWidth).Val,
      Result.Val, Offset);
    Offset := Offset + sz;
  end;

  Result := Transpose(Result, [1, 0, 2]).Reshape([Channels * FilterH *
    FilterW, m * ConvOutHeight * ConvOutWidth]);
end;

procedure BackwardAdd(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + ReduceTo(ADy, arr[0].Data);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + ReduceTo(ADy, arr[1].Data);
end;

procedure BackwardConv2D(arr: TVariableArr; ADy: TTensor);
var
  DyReshaped, dX, dW, dXCol, wReshape: TTensor;
  i: longint;
begin
  { Note the passed sequence from forward pass:
    arr -->[X, w, PaddingHeight, PaddingWidth, StrideHeight, StrideWidth, XCol],
            0  1        2             3             4            5         6}

  DyReshaped := Transpose(ADy, [1, 2, 3, 0]).Reshape([arr[1].Shape[0],
    ADy.Size div arr[1].Shape[0]]);
  wReshape := arr[1].Data.Reshape([arr[1].Shape[0], arr[1].Shape[1] *
    arr[1].Shape[2] * arr[1].Shape[3]]);
  dXCol := wReshape.T.Dot(DyReshaped);
  dX := Col2ImBatch(dXCol,
    arr[0].Shape[1], arr[0].Shape[2], arr[0].Shape[3],
    arr[1].Shape[2], arr[1].Shape[3],
    round(arr[2].Data.Val[0]), round(arr[3].Data.Val[0]),
    round(arr[4].Data.Val[0]), round(arr[5].Data.Val[0]));

  dW := DyReshaped.Dot(arr[6].Data.T).Reshape(arr[1].Shape);

  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + dX;
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + dW;
end;

procedure BackwardDivide(arr: TVariableArr; ADy: TTensor);
var
  A, B: TTensor;
begin
  if arr[0].RequiresGrad then
  begin
    A := ADy / arr[1].Data;
    arr[0].Grad := arr[0].Grad + ReduceTo(A, arr[0].Data);
  end;
  if arr[1].RequiresGrad then
  begin
    B := -ADy * arr[0].Data / arr[1].Data ** 2;
    arr[1].Grad := arr[1].Grad + ReduceTo(B, arr[1].Data);
  end;
end;

procedure BackwardSubtract(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + ReduceTo(ADy, arr[0].Data);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad - ReduceTo(ADy, arr[1].Data);
end;

procedure BackwardMultiply(arr: TVariableArr; ADy: TTensor);
var
  B, A: TTensor;
begin
  if arr[0].RequiresGrad then
  begin
    A := noe.Math.Multiply(ADy, arr[1].Data);
    arr[0].Grad := arr[0].Grad + ReduceTo(A, arr[0].Data);
  end;
  if arr[1].RequiresGrad then
  begin
    B := noe.Math.Multiply(ADy, arr[0].Data);
    arr[1].Grad := arr[1].Grad + ReduceTo(B, arr[1].Data);
  end;
end;

procedure BackwardMultiplyC(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + noe.Math.Multiply(ADy, arr[1].Data);
end;

procedure BackwardMatmul(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + MatMul(ADy, arr[1].Data.T);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + MatMul(arr[0].Data.T, ADy);
end;

procedure BackwardCosh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Sinh(arr[0].Data));
end;

procedure BackwardReshape(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + ADy.Reshape(arr[0].Shape);
end;

procedure BackwardSigmoid(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (Sigmoid(arr[0].Data) * (1 - Sigmoid(arr[0].Data)));
end;

procedure BackwardSinh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Cosh(arr[0].Data));
end;

procedure BackwardSqr(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * 2 * arr[0].Data);
end;

procedure BackwardMeanElement(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * Ones(arr[0].Data.Shape));
end;

procedure BackwardSqrt(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * 0.5 * 1 / (arr[0].Data ** 0.5));
end;

procedure BackwardNegate(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad - ADy;
end;

procedure BackwardReLU(arr: TVariableArr; ADy: TTensor);
var
  i: longint;
begin
  if arr[0].RequiresGrad then
    for i := 0 to Length(arr[0].Data.Val) - 1 do
      if arr[0].Data.Val[i] > 0 then
        arr[0].Grad.Val[i] := arr[0].Grad.Val[i] + ADy.Val[i];
end;

procedure BackwardLeakyReLU(arr: TVariableArr; ADy: TTensor);
var
  i: longint;
begin
  if arr[0].RequiresGrad then
    for i := 0 to Length(arr[0].Data.Val) - 1 do
      if arr[0].Data.Val[i] > 0 then
        arr[0].Grad.Val[i] := arr[0].Grad.Val[i] + ADy.Val[i]
      else
        { arr[1].Data.Val[0] refers to v parameter in LeakyReLU }
        arr[0].Grad.Val[i] := arr[0].Grad.Val[i] + ADy.Val[i] * arr[1].Data.Val[0];
end;

procedure BackwardLn(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy / arr[0].Data);
end;

procedure BackwardExp(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Exp(arr[0].Data));
end;

procedure BackwardTanh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy / noe.Math.Cosh(arr[0].Data) ** 2);
end;

procedure BackwardMax(arr: TVariableArr; ADy: TTensor);
var
  maxval: NFloat;
  A: TTensor;
  i: integer;
begin
  if arr[0].RequiresGrad then
  begin
    A      := Zeros(Arr[0].Data.Shape);
    maxval := MaxValue(arr[0].Data.Val);
    for i := 0 to length(arr[0].Data.Val) - 1 do
      if arr[0].Data.Val[i] = maxval then
        A.Val[i] := 1;
    arr[0].Grad  := arr[0].Grad + A;
  end;
end;

procedure BackwardMean(arr: TVariableArr; ADy: TTensor);
var
  szArr, szDy: longint;
begin
  szArr := ShapeToSize(arr[0].Data.Shape);
  szDy  := ShapeToSize(ADy.Shape);
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + CreateTensor(arr[0].Data.Shape,
      ADy.Val[0] / (szArr / szDy));
end;

procedure BackwardSum(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + CreateTensor(arr[0].Data.Shape, ADy.Val[0]);
end;

function Einsum(Subscripts: string; Pots: array of TTensor): TTensor;
type
  TNameDimsMap = specialize TFPGMap<string, TIntVector>;
  TStringIntMap = specialize TFPGMap<string, longint>;
  TIntVectorArr = array of TIntVector;
var
  re: TRegExpr;
  match, keepGoing, skipCombo: boolean;
  i, j, len: longint;
  split, tables: TStringArray;
  broadcastList, flatTables, originalTables, uniqueTables: ansistring;
  nameAndDims: TNameDimsMap;
  uniqueDict: TStringIntMap;
  comb, bcomb, indices, flatDims, broadcastDims, combinations: TIntVector;
  forMultiPlying: TFloatVector;
  dims, combos, broadcastCombos: TIntVectorArr;
  s: string;
  plug, Value, v: NFloat;

  function Combo(dimension: array of longint): TIntVectorArr;
  var
    row, res: TIntVector;
    tmpResult: TIntVectorArr;

    procedure iterate(d: longint; shape, res: array of longint);
    var
      i, j: longint;
    begin
      if d >= Length(dimension) then
      begin
        SetLength(row, Length(res));
        for j := 0 to Length(res) - 1 do
          row[j] := res[j];
        SetLength(tmpResult, Length(tmpResult) + 1);
        tmpResult[Length(tmpResult) - 1] := row;
        exit;
      end;

      for i := 0 to shape[d] - 1 do
      begin
        res[d] := i;
        iterate(d + 1, shape, res);
      end;
    end;

  begin
    SetLength(tmpResult, 0);
    SetLength(res, Length(dimension));
    iterate(0, dimension, res);

    Result := tmpResult;
  end;

begin
  if '->' in Subscripts then
  begin
    re    := TRegExpr.Create('(.)\1');
    match := re.Exec(Subscripts);
    { there are repeated letters, return diagonal }
    if match then
    begin
      Assert(Pots[0].Shape[0] = Pots[0].Shape[1], 'Cannot collapse index ' +
        re.Match[0].Chars[0]);

      len := Pots[0].Shape[0];


      SetLength(Result.Val, len);
      Result.ReshapeInplace([len]);
      for i := 0 to len - 1 do
        Result.Val[i] := Pots[0].GetAt([i, i]).Val[0];
    end

    { tensor dot multiplication and specific dimension broadcasting }
    else
    begin
      split  := Subscripts.Split('->');
      tables := split[0].Split(',');
      broadcastList := split[2];

      nameAndDims := TNameDimsMap.Create;
      SetLength(dims, Length(Pots));
      for i := 0 to length(Pots) - 1 do
      begin
        nameAndDims.Add(tables[i], Pots[i].Shape);
        dims[i] := Pots[i].Shape;
      end;

      SetLength(flatDims, 0);
      flatTables     := '';
      originalTables := '';
      for i := 0 to Length(dims) - 1 do
      begin
        for s in tables[i] do
        begin
          flatTables := flatTables + s;
          if not (s in originalTables) then
            originalTables := originalTables + s;
        end;

        for j := 0 to length(dims[i]) - 1 do
        begin
          SetLength(flatDims, Length(flatDims) + 1);
          flatDims[Length(flatDims) - 1] := dims[i][j];
        end;
      end;
      uniqueTables := SortStr(originalTables);

      uniqueDict := TStringIntMap.Create;
      for i := 0 to Length(flatTables) - 1 do
        uniqueDict.Add(flatTables.Chars[i], flatDims[i]);

      SetLength(combinations, 0);
      for s in uniqueTables do
        if uniqueDict.IndexOf(s) > -1 then
        begin
          SetLength(combinations, Length(combinations) + 1);
          combinations[Length(combinations) - 1] := uniqueDict.KeyData[s];
        end;
      keepGoing := True;

      setLength(broadcastDims, 0);
      while keepGoing do
        for s in broadcastList do
        begin
          setLength(broadcastDims, length(broadcastDims) + 1);
          broadcastDims[length(broadcastDims) - 1] := uniqueDict.KeyData[s];
          keepGoing := False;
        end;

      combos := combo(combinations);
      broadcastCombos := combo(broadcastDims);

      Result := CreateTensor(broadcastDims, 0.0);

      for bcomb in broadcastCombos do
      begin
        plug := 0;
        for comb in combos do
        begin
          skipCombo := False;

          { TODO optimize these lines to obtain skipCombo}
          for s in broadcastList do
            if comb[uniqueTables.IndexOf(s)] <> bcomb[broadcastList.IndexOf(s)] then
              skipCombo := True;

          if not skipCombo then
          begin
            SetLength(forMultiPlying, Length(tables));
            for i := 0 to Length(tables) - 1 do
            begin
              setlength(indices, Length(tables[i]));
              for j := 0 to length(tables[i]) - 1 do
                indices[j] := comb[uniqueTables.IndexOf(tables[i].Chars[j])];

              forMultiPlying[i] := (Pots[i].GetAt(indices).Val[0]);
            end;

            Value := 1;
            for v in forMultiPlying do
              Value := Value * v;

            plug := plug + Value;
          end;
        end;
        Result.Val[IndexToOffset(bcomb, broadcastDims)] := plug;
      end;
    end;
  end

  { there are repeated letters but no '->', return sum of diagonal }
  else
    Result := Math.sum(Einsum(Subscripts + '->', Pots).Val);
end;

function ApplyUfunc(A: TTensor; Func: TUFunc): TTensor;
var
  i: longint;
begin
  Result.ReshapeInplace(A.Shape);
  SetLength(Result.val, Length(A.val));
  for i := 0 to length(A.val) - 1 do
    Result.val[i] := func(A.val[i]);
end;

function ApplyBfunc(A, B: TTensor; Func: TBFunc): TTensor;
var
  i: longint;
  A_bcast, B_bcast: TTensor;
  outdim: TIntVector;
begin
  { if the dimensions are the same, perform usual element-wise operation }
  if IntVectorEquals(A.Shape, B.Shape) then
  begin
    { ---------- If you can BLAS it, BLAS it ---------- }
    if (Func = @Add_F) and IsBlasfuncAvailable(blas_daxpy) then
      exit(Add_BLAS(A, B));

    { ---------- Otherwise, go vanilla ---------- }
    Result.ReshapeInplace(A.Shape);
    SetLength(Result.Val, Length(A.Val));
    for i := 0 to Length(A.Val) - 1 do
      Result.Val[i] := Func(A.Val[i], B.Val[i]);
  end
  else { otherwise, perform broadcasting }
  begin
    { first, check if broadcastable }
    Assert(IsBroadcastable(A, B), 'Cannot perform broadcasting');

    { If either one is a scalar, i.e., A.Size=1 or B.Size=1 }
    if (B.Size = 1) then
    begin
      { tensor-scalar broadcasting. }
      Result := CreateEmptyTensor(A.Shape);
      for i := 0 to Length(A.Val) - 1 do
        Result.Val[i] := Func(A.Val[i], B.Val[0]);
      exit;
    end;

    if (A.Size = 1) then
    begin
      { scalar-tensor broadcasting. }
      Result := CreateEmptyTensor(B.Shape);
      for i := 0 to Length(B.Val) - 1 do
        Result.Val[i] := Func(A.Val[0], B.Val[i]);
      exit;
    end;

    { Current general broadcasting implementation seems slow. At least, for a
      specific rank-2 tensor case, i.e., A.Ndims = B.Ndims = 2 go for hand-crafted
      optimization. The workaround is to make a copy of orginal tensor, then
      "tile" it to match the output shape. The trade-off is storage complexity. }
    if (A.NDims = 2) and (B.NDims = 2) then
    begin
      { 2-tensor-2-tenspr tensor broadcast bfunc. }
      outdim := GetBroadcastDims(a, b);

      // handle A
      if A.Shape[0] < outdim[0] then
        A_bcast := TileRow(A, outdim[0])
      else if A.Shape[1] < outdim[1] then
        A_bcast := TileColumn(A, outdim[1])
      else
        A_bcast := A;

      // handle B
      if B.Shape[0] < outdim[0] then
        B_bcast := TileRow(B, outdim[0])
      else if B.Shape[1] < outdim[1] then
        B_bcast := TileColumn(B, outdim[1])
      else
        B_bcast := B;
      exit(ApplyBfunc(A_bcast, B_bcast, Func));
      FreeAndNil(A_bcast);
      FreeAndNil(B_bcast);
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
end;

function IsBlasfuncAvailable(Func: Pointer): boolean;
begin
  Result := (NoeConfig.useBLAS) and (Func <> nil);
end;


end.
