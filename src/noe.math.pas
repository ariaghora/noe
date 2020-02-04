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

{$mode objfpc}{$H+}

interface

uses
  Classes, fgl, Math, noe, noe.backend.blas, noe.backend.native, noe.utils,
  RegExpr, strutils, SysUtils;

type
  { Wrapping FPC's f:R->R unary functions in math unit }
  TUFunc = function(v: float): float;

  { Wrapping FPC's f:RxR->R binary functions in math unit }
  TBFunc = function(v1, v2: double): double;

{ Helper to apply a function on each tensor's element }
function ApplyUfunc(A: TTensor; Func: TUFunc): TTensor; inline;
function ApplyBfunc(A, B: TTensor; Func: TBFunc): TTensor; inline;

{ Some of functions belong to system unit are in different format. Hence, they
  need to be wrapped to make them compatible. They are given suffix "F"
  (indicating double-valued function) to avoid confusion. }
function Sin_F(x: double): double;
function Cos_F(x: double): double;
function Exp_F(x: double): double;
function Ln_F(x: double): double;
function AddF(v1, v2: double): double; inline;
function SubtractF(v1, v2: double): double; inline;
function DivideF(v1, v2: double): double; inline;
function MultiplyF(v1, v2: double): double; inline;

{ TTensor math ----------------------------------------------------------------}

{ binary functions for tensors }
function Add(A, B: TTensor): TTensor; inline;
function Subtract(A, B: TTensor): TTensor; inline;
function Divide(A, B: TTensor): TTensor; inline;
function Multiply(A, B: TTensor): TTensor; inline;
function MatMul(A, B: TTensor): TTensor; inline;

{ unary functions for tensors }
function ArgMax(M: TTensor): TTensor; inline; overload;
function ArgMax(M: TTensor; axis: byte): TTensor; inline; overload;
function Max(M: TTensor): TTensor; inline;
function Max(M: TTensor; axis: byte): TTensor; inline; overload;
function Mean(M: TTensor): TTensor; inline;
function Mean(M: TTensor; axis: byte): TTensor; inline; overload;
function DegToRad(A: TTensor): TTensor;
function RadToDeg(A: TTensor): TTensor;
function Cos(A: TTensor): TTensor;
function Cosh(A: TTensor): TTensor;
function Exp(A: TTensor): TTensor;
function Log10(A: TTensor): TTensor;
function Log2(A: TTensor): TTensor;
function Log(A: TTensor): TTensor;
function Tan(A: TTensor): TTensor;
function Tanh(A: TTensor): TTensor;
function Power(A: TTensor; exponent: double): TTensor; overload;
function Power(A, B: TTensor): TTensor; overload;
function ReLU(T: TTensor): TTensor;
function Sin(A: TTensor): TTensor;
function Sinh(A: TTensor): TTensor;
function SoftMax(A: TTensor; axis: byte): TTensor; inline;
function Sum(M: TTensor): TTensor; inline;
function Sum(M: TTensor; axis: byte): TTensor; inline; overload;

{ Evaluates the Einstein summation convention on the operands. Very slow now.
  The initial implementation is heavily inspired from Kyle Hundman's attempt to
  mirror numpy's einsum, so not all operations are supported. Any helps are
  welcome. }
function Einsum(Subscripts: string; Pots: array of TTensor): TTensor;

function Transpose2D(T: TTensor): TTensor;
function Transpose(T: TTensor; dims: array of longint): TTensor;
function Transpose(T: TTensor): TTensor;

 { TVariable math --------------------------------------------------------------}
 { forward mode }
function Add(A, B: TVariable): TVariable;
function Divide(A, B: TVariable): TVariable;
function Subtract(A, B: TVariable): TVariable;
function Multiply(A, B: TVariable): TVariable;
function MultiplyC(A: TVariable; x: double): TVariable;
function MatMul(A, B: TVariable): TVariable;

function Negate(A: TVariable): TVariable;
function Cosh(A: TVariable): TVariable;
function Log(A: TVariable): TVariable;
function Sinh(A: TVariable): TVariable;
function Sqr(A: TVariable): TVariable;
function Sqrt(A: TVariable): TVariable;
function ReLU(A: TVariable): TVariable;
function Tanh(A: TVariable): TVariable;
function Exp(A: TVariable): TVariable;
function Max(A: TVariable): TVariable;
function Max(A: TVariable; axis: byte): TVariable; overload;
function Mean(A: TVariable; axis: byte): TVariable;
function Mean(A: TVariable): TVariable; overload;
function Sum(A: TVariable; axis: byte): TVariable;
function Sum(A: TVariable): TVariable; overload;

{ backward mode }
procedure BackwardAdd(arr: TVariableArr; ADy: TTensor);
procedure BackwardDivide(arr: TVariableArr; ADy: TTensor);
procedure BackwardSubtract(arr: TVariableArr; ADy: TTensor);
procedure BackwardMultiply(arr: TVariableArr; ADy: TTensor);
procedure BackwardMultiplyC(arr: TVariableArr; ADy: TTensor);
procedure BackwardMatmul(arr: TVariableArr; ADy: TTensor);

procedure BackwardCosh(arr: TVariableArr; ADy: TTensor);
procedure BackwardLn(arr: TVariableArr; ADy: TTensor);
procedure BackwardExp(arr: TVariableArr; ADy: TTensor);
procedure BackwardMax(arr: TVariableArr; ADy: TTensor);
procedure BackwardMean(arr: TVariableArr; ADy: TTensor);
procedure BackwardNegate(arr: TVariableArr; ADy: TTensor);
procedure BackwardReLU(arr: TVariableArr; ADy: TTensor);
procedure BackwardSinh(arr: TVariableArr; ADy: TTensor);
procedure BackwardSqr(arr: TVariableArr; ADy: TTensor);
procedure BackwardSqrt(arr: TVariableArr; ADy: TTensor);
procedure BackwardSum(arr: TVariableArr; ADy: TTensor);
procedure BackwardTanh(arr: TVariableArr; ADy: TTensor);

{ aggregate functions, derived from above functions }
function SoftMax(A: TVariable; axis: byte): TVariable;

{ If target is the result of broadcasting, reduce to its original shape }
function ReduceTo(Target, Other: TTensor): TTensor;

implementation

function AddF(v1, v2: double): double;
begin
  Result := v1 + v2;
end;

function SubtractF(v1, v2: double): double;
begin
  Result := v1 - v2;
end;

function DivideF(v1, v2: double): double;
begin
  Result := v1 / v2;
end;

function MultiplyF(v1, v2: double): double;
begin
  Result := v1 * v2;
end;

function Add(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @AddF);
end;

function Subtract(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @SubtractF);
end;

function Divide(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @DivideF);
end;

function Multiply(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @MultiplyF);
end;

function MatMul(A, B: TTensor): TTensor;
begin
  Assert((length(A.Shape) <= 2) and (length(B.Shape) <= 2),
    'Tensor dimension must be <= 2.');

  { calculates matrix multiplication according to the backend }
  if noe.NoeConfig.useBLAS then
    Result := MatMul_BLAS(A, B)
  else
    Result := MatMul_Native(A, B);
end;

function ArgMax(M: TTensor): TTensor;
begin
  Result := TTensor.Create;
  SetLength(Result.Val, 1);
  Result.Val[0] := ArgMax(M.Val);
  Result.Reshape([1]);
end;

function ArgMax(M: TTensor; axis: byte): TTensor;
var
  i: integer;
begin
  Assert(Length(M.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Assert(axis in [0, 1], MSG_ASSERTION_INVALID_AXIS);
  Result := TTensor.Create;
  if axis = 0 then
  begin
    SetLength(Result.Val, M.Shape[1]);
    Result.Reshape([1, M.Shape[1]]);
    for i := 0 to M.Shape[1] - 1 do
      Result.Val[i] := ArgMax(GetColumn(M, i).Val);
  end
  else
  begin
    SetLength(Result.Val, M.Shape[0]);
    Result.Reshape([M.Shape[0], 1]);
    for i := 0 to M.Shape[0] - 1 do
      Result.Val[i] := ArgMax(GetRow(M, i).Val);
  end;
end;

function Max(M: TTensor): TTensor;
begin
  Result := TTensor.Create;
  SetLength(Result.Val, 1);
  Result.Val[0] := MaxValue(M.Val);
  Result.Reshape([1]);
end;

function Max(M: TTensor; axis: byte): TTensor;
var
  i: longint;
begin
  Assert(Length(M.Shape) = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Assert(axis in [0, 1], MSG_ASSERTION_INVALID_AXIS);
  Result := TTensor.Create;
  if axis = 0 then
  begin
    SetLength(Result.Val, M.Shape[1]);
    Result.Reshape([1, M.Shape[1]]);
    for i := 0 to M.Shape[1] - 1 do
      Result.Val[i] := MaxValue(GetColumn(M, i).Val);
  end
  else
  begin
    SetLength(Result.Val, M.Shape[0]);
    Result.Reshape([M.Shape[0], 1]);
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
    if noe.NoeConfig.useBLAS then
      Result := MeanCol_BLAS(M)
    else
      Result := MeanCol_Native(M);
  end
  else
  if noe.NoeConfig.useBLAS then
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
    if noe.NoeConfig.useBLAS then
      Result := SumCol_BLAS(M)
    else
      Result := SumCol_Native(M);
  end
  else
  if noe.NoeConfig.useBLAS then
    Result := SumRow_BLAS(M)
  else
    Result := SumRow_Native(M);
end;

function DegToRad(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.degtorad);
end;

function RadToDeg(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.radtodeg);
end;

function Log10(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.log10);
end;

function Log2(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.log2);
end;

function Log(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Ln_F);
end;

function Transpose2D(T: TTensor): TTensor;
var
  i, j: longint;
begin
  Assert(Length(T.Shape) = 2, 'Transpose2D only accepts rank-2 tensors');
  Result := TTensor.Create;
  Result.Reshape([T.Shape[1], T.Shape[0]]);
  SetLength(Result.Val, Length(T.Val));
  for i := 0 to T.Shape[0] - 1 do
    for j := 0 to T.Shape[1] - 1 do
      Result.Val[j * T.Shape[0] + i] := T.Val[i * T.Shape[1] + j];
end;

function Transpose(T: TTensor; dims: array of longint): TTensor;
var
  resultedIdx, dimsLetter: string;
  i: longint;
begin
  dimsLetter := DimsToLetter(dims);
  Assert(Length(dims) = length(T.Shape),
    'dims length does not match tensor dimension');
  resultedIdx := DimsToLetter(dims);
  for i := 0 to Length(dims) - 1 do
    resultedIdx[i + 1] := dimsLetter.Chars[dims[i]];
  Result := Einsum(dimsLetter + '->' + resultedIdx, [T]);
end;

function Transpose(T: TTensor): TTensor;
begin
  // attempt with 2d transpose first
  if (Length(T.Shape) = 2) then
    Result := Transpose2D(T)
  else
    Result := Einsum(DimsToLetter(T.Shape) + '->' +
      ReverseString(DimsToLetter(T.Shape)), [T]);
end;

function ReLU(T: TTensor): TTensor;
var
  i: longint;
begin
  Result := TTensor.Create;
  Result.Reshape(T.Shape);
  SetLength(Result.Val, Length(T.Val));
  for i := 0 to Length(Result.Val) - 1 do
    Result.Val[i] := Max(0, T.Val[i]);
end;

function Sin_F(x: double): double;
begin
  Result := System.Sin(x);
end;

function Cos_F(x: double): double;
begin
  Result := System.Cos(x);
end;

function Exp_F(x: double): double;
begin
  Result := System.exp(x);
end;

function Ln_F(x: double): double;
begin
  Result := system.ln(x);
end;

function Sin(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Sin_F);
end;

function Sinh(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.sinh);
end;

function Cos(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Cos_F);
end;

function Cosh(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.cosh);
end;

function Tan(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.tan);
end;

function Tanh(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.tanh);
end;

function Power(A: TTensor; exponent: double): TTensor;
begin
  Result := ApplyBfunc(A, exponent, @Math.power);
end;

function Power(A, B: TTensor): TTensor;
begin
  Result := ApplyBfunc(A, B, @Math.power);
end;

function Exp(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Exp_F);
end;

function Add(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data + B.Data, 'Add', @BackwardAdd);
  Result.RequiresGrad := True;
  Result.AddPrev([A, B]);
end;

function Divide(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data / B.Data, 'Divide', @BackwardDivide);
  Result.RequiresGrad := True;
  Result.AddPrev([A, B]);
end;

function Subtract(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data - B.Data, 'Subtract', @BackwardSubtract);
  Result.RequiresGrad := True;
  Result.AddPrev([A, B]);
end;

function Multiply(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Multiply(A.Data, B.Data),
    'Multiply', @BackwardMultiply);
  Result.RequiresGrad := True;
  Result.AddPrev([A, B]);
end;

function MultiplyC(A: TVariable; x: double): TVariable;
begin
  Result := TVariable.Create(noe.Math.Multiply(A.Data, x), 'MultiplyC',
    @BackwardMultiplyC);
  Result.RequiresGrad := True;

  SetLength(Result.Prev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := TVariable.Create(x, '');
  Result.Prev[1].RequiresGrad := False;
end;

function MatMul(A, B: TVariable): TVariable;
begin
  Assert(A.Shape[1] = B.Shape[0], MSG_ASSERTION_DIM_MISMATCH);
  Result := TVariable.Create(noe.Math.MatMul(A.Data, B.Data), 'MatMul', @BackwardMatmul);
  Result.RequiresGrad := True;
  Result.AddPrev([A, B]);
end;

function Negate(A: TVariable): TVariable;
begin
  Result := TVariable.Create(-A.Data, 'Negate', @BackwardNegate);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Cosh(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Cosh(A.Data), 'Cosh', @BackwardCosh);
  Result.RequiresGrad := True;

  Result.AddPrev(A);
end;

function Log(A: TVariable): TVariable;
begin
  Result := TVariable.Create(Log(A.Data), 'Ln', @BackwardLn);
  Result.RequiresGrad := True;

  Result.AddPrev(A);
end;

function Sinh(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Sinh(A.Data), 'Sinh', @BackwardSinh);
  Result.RequiresGrad := True;

  Result.AddPrev(A);
end;

function Sqr(A: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data ** 2, 'Sqr', @BackwardSqr);
  Result.RequiresGrad := True;

  Result.AddPrev(A);
end;

function Sqrt(A: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data ** 0.5, 'Sqrt', @BackwardSqrt);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function ReLU(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.ReLU(A.Data), 'ReLU', @BackwardReLU);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Tanh(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Tanh(A.Data), 'Tanh', @BackwardTanh);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Exp(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Exp(A.Data), 'Exp', @BackwardExp);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Max(A: TVariable): TVariable;
begin
  Result := TVariable.Create(Max(A.Data), 'Max', @BackwardMax);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Max(A: TVariable; axis: byte): TVariable;
begin
  { Max along axis has no gradient (?) }
  Result := TVariable.Create(Max(A.Data, axis), 'Max');
end;

function Mean(A: TVariable; axis: byte): TVariable;
begin
  Result := TVariable.Create(Mean(A.Data, axis), 'Mean', @BackwardMean);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Mean(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Mean(A.Data), 'Mean', @BackwardMean);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Sum(A: TVariable; axis: byte): TVariable;
begin
  Result := TVariable.Create(noe.Math.sum(A.Data, axis), 'Sum', @BackwardSum);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function Sum(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.sum(A.Data), 'Sum', @BackwardSum);
  Result.RequiresGrad := True;
  Result.AddPrev(A);
end;

function SoftMax(A: TVariable; axis: byte): TVariable;
var
  X, Y: TVariable;
begin
  X      := A - Max(A, axis);
  Y      := Exp(X);
  Result := Y / sum(Y, axis);
end;

function ReduceTo(Target, Other: TTensor): TTensor;
var
  ax, i: longint;
begin
  Result := Target;
  ax     := -1;
  for i := 0 to Length(Target.Shape) - 1 do
    if Target.Shape[i] > Other.Shape[i] then
      ax := i;
  if ax > -1 then
    Result := Sum(Target, ax);
end;

procedure BackwardAdd(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + ReduceTo(ADy, arr[0].Data);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + ReduceTo(ADy, arr[1].Data);
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
    arr[0].Grad := arr[0].Grad + noe.Math.MatMul(ADy, arr[1].Data.T);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + noe.Math.MatMul(arr[0].Data.T, ADy);
end;

procedure BackwardCosh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Sinh(arr[0].Data));
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
  maxval: double;
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
  plug, Value, v: double;

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

      Result := TTensor.Create;
      len    := Pots[0].Shape[0];
      SetLength(Result.Val, len);
      Result.Reshape([len]);
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
  Result := TTensor.Create;
  Result.Reshape(A.Shape);
  SetLength(Result.val, Length(A.val));
  for i := 0 to length(A.val) - 1 do
    Result.val[i] := func(A.val[i]);
end;

function ApplyBfunc(A, B: TTensor; Func: TBFunc): TTensor;
var
  i: longint;
  br: TBroadcastResult;
begin
  { if the dimensions are the same, perform usual element-wise operation }
  if IntVectorEquals(A.Shape, B.Shape) then
  begin
    Result := TTensor.Create;
    Result.Reshape(A.Shape);

    SetLength(Result.Val, Length(A.Val));
    for i := 0 to Length(A.Val) - 1 do
      Result.Val[i] := Func(A.Val[i], B.Val[i]);
  end
  else { otherwise, perform broadcasting }
  begin
    { first, check if broadcastable }
    Assert(IsBroadcastable(A, B), 'Cannot perform broadcasting');

    { Current general broadcasting implementation seems slow. At least, for a
      specific rank-2 tensor case, go for optimization. The workaround is to
      make a copy of orginal tensor, then "tile" it to match the output shape.
      The trade-off is storage complexity. }

    br := Broadcast(A, B);

    Result := TTensor.Create;
    Result.Reshape(br.broadcastShape);
    SetLength(Result.Val, ShapeToSize(br.broadcastShape));
    { apply binary function }
    for i := 0 to ShapeToSize(br.broadcastShape) - 1 do
      Result.Val[i] := Func(br.A.Val[i], br.B.Val[i]);
  end;
end;

end.
