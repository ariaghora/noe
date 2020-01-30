{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit contains required basic operations for automatic gradient
 computation.

 To do:
  - Apply ReduceTo on the op that involves broadcasting
}

unit noe.op.base;

{$mode objfpc}

interface

uses
  Classes, SysUtils, noe.core, noe.Math, noe.autograd;

{ forward functions -----------------------------------------------------------}
function Add(A, B: TVariable): TVariable;
function Divide(A, B: TVariable): TVariable;
function Subtract(A, B: TVariable): TVariable;
function Multiply(A, B: TVariable): TVariable;
function MultiplyC(A: TVariable; x: double): TVariable;
function MatMul(A, B: TVariable): TVariable;

function Negate(A: TVariable): TVariable;
function Cosh(A: TVariable): TVariable;
function Sinh(A: TVariable): TVariable;
function Sqr(A: TVariable): TVariable;
function Sqrt(A: TVariable): TVariable;
function ReLU(A: TVariable): TVariable;
function Tanh(A: TVariable): TVariable;
function Exp(A: TVariable): TVariable;
function Mean(A: TVariable; axis: byte): TVariable;
function Mean(A: TVariable): TVariable; overload;
function Sum(A: TVariable; axis: byte): TVariable;
function Sum(A: TVariable): TVariable; overload;

{ backward functions ----------------------------------------------------------}
procedure BwAdd(arr: TVariableArr; ADy: TTensor);
procedure BwDivide(arr: TVariableArr; ADy: TTensor);
procedure BwSubtract(arr: TVariableArr; ADy: TTensor);
procedure BwMultiply(arr: TVariableArr; ADy: TTensor);
procedure BwMultiplyC(arr: TVariableArr; ADy: TTensor);
procedure BwMatmul(arr: TVariableArr; ADy: TTensor);

procedure BwCosh(arr: TVariableArr; ADy: TTensor);
procedure BwExp(arr: TVariableArr; ADy: TTensor);
procedure BwMean(arr: TVariableArr; ADy: TTensor);
procedure BwNegate(arr: TVariableArr; ADy: TTensor);
procedure BwReLU(arr: TVariableArr; ADy: TTensor);
procedure BwSinh(arr: TVariableArr; ADy: TTensor);
procedure BwSqr(arr: TVariableArr; ADy: TTensor);
procedure BwSqrt(arr: TVariableArr; ADy: TTensor);
procedure BwSum(arr: TVariableArr; ADy: TTensor);
procedure BwTanh(arr: TVariableArr; ADy: TTensor);

{ aggregate functions, derived from above functions ---------------------------}
function SoftMax(A: TVariable; axis: byte): TVariable;

{ If target is the result of broadcasting, reduce to its original shape }
function ReduceTo(Target, Other: TTensor): TTensor;

operator := (Val: double) V: TVariable;
operator +(A, B: TVariable) C: TVariable;
operator -(A, B: TVariable) C: TVariable;
operator -(A: TVariable) B: TVariable;
operator * (A, B: TVariable) C: TVariable;


implementation

function Add(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data + B.Data, 'Add', @BwAdd);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := B;
end;

function Divide(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data / B.Data, 'Divide', @BwDivide);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := B;
end;

function Subtract(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data - B.Data, 'Subtract', @BwSubtract);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := B;
end;

function Multiply(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Multiply(A.Data, B.Data),
    'Multiply', @BwMultiply);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := B;
end;

function MultiplyC(A: TVariable; x: double): TVariable;
begin
  Result := TVariable.Create(noe.Math.Multiply(A.Data, x), 'MultiplyC', @BwMultiplyC);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := TVariable.Create(x, '');
  Result.Prev[1].RequiresGrad := False;
end;

function MatMul(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.MatMul(A.Data, B.Data), 'MatMul', @BwMatmul);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := B;
end;

function Negate(A: TVariable): TVariable;
begin
  Result := TVariable.Create(-A.Data, 'Negate', @BwNegate);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Cosh(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Cosh(A.Data), 'Cosh', @BwCosh);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Sinh(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Sinh(A.Data), 'Sinh', @BwSinh);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Sqr(A: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data ** 2, 'Sqr', @BwSqr);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Sqrt(A: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data ** 0.5, 'Sqrt', @BwSqrt);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function ReLU(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.ReLU(A.Data), 'ReLU', @BwReLU);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Tanh(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Tanh(A.Data), 'Tanh', @BwTanh);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Exp(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Exp(A.Data), 'Exp', @BwExp);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Mean(A: TVariable; axis: byte): TVariable;
begin
  Result := TVariable.Create(noe.Math.Mean(A.Data, axis), 'Mean', @BwMean);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Mean(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Mean(A.Data), 'Mean', @BwMean);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Sum(A: TVariable; axis: byte): TVariable;
begin
  Result := TVariable.Create(noe.Math.sum(A.Data, axis), 'Sum', @BwSum);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

function Sum(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.sum(A.Data), 'Sum', @BwSum);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

procedure BwAdd(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + ReduceTo(ADy, arr[0].Data);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + ReduceTo(ADy, arr[1].Data);
end;

function SoftMax(A: TVariable; axis: byte): TVariable;
var
  X, Y: TVariable;
begin
  X := exp(A);
  Y := Divide(exp(A), sum(X, axis));
  Result := Y;
end;

function ReduceTo(Target, Other: TTensor): TTensor;
var
  ax, i: longint;
begin
  Result := Target;
  ax := -1;
  for i := 0 to Length(Target.Shape) - 1 do
    if Target.Shape[i] > Other.Shape[i] then
      ax := i;
  if ax > -1 then
    Result := Sum(Target, ax);
end;

procedure BwDivide(arr: TVariableArr; ADy: TTensor);
var
  A, B: TTensor;
  i: longint;
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

procedure BwSubtract(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + ReduceTo(ADy, arr[0].Data);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad - ReduceTo(ADy, arr[1].Data);
end;

procedure BwMultiply(arr: TVariableArr; ADy: TTensor);
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

procedure BwMultiplyC(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + noe.Math.Multiply(ADy, arr[1].Data);
end;

procedure BwMatmul(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + noe.Math.MatMul(ADy, arr[1].Data.T);
  if arr[1].RequiresGrad then
    arr[1].Grad := arr[1].Grad + noe.Math.MatMul(arr[0].Data.T, ADy);
end;

procedure BwCosh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Sinh(arr[0].Data));
end;

procedure BwSinh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Cosh(arr[0].Data));
end;

procedure BwSqr(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * 2 * arr[0].Data);
end;

procedure BwMeanElement(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * Ones(arr[0].Data.Shape));
end;

procedure BwSqrt(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * 0.5 * 1 / (arr[0].Data ** 0.5));
end;

procedure BwNegate(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad - ADy;
end;

procedure BwReLU(arr: TVariableArr; ADy: TTensor);
var
  i: longint;
begin
  if arr[0].RequiresGrad then
    for i := 0 to Length(arr[0].Data.Val) - 1 do
      if arr[0].Data.Val[i] > 0 then
        arr[0].Grad.Val[i] := arr[0].Grad.Val[i] + ADy.Val[i];
end;

procedure BwExp(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy * noe.Math.Exp(arr[0].Data));
end;

procedure BwTanh(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + (ADy / noe.Math.Cosh(arr[0].Data) ** 2);
end;

procedure BwMean(arr: TVariableArr; ADy: TTensor);
var
  szArr, szDy: longint;
begin
  szArr := ShapeToSize(arr[0].Data.Shape);
  szDy := ShapeToSize(ADy.Shape);
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + CreateTensor(arr[0].Data.Shape,
      ADy.Val[0] / (szArr / szDy));
end;

procedure BwSum(arr: TVariableArr; ADy: TTensor);
begin
  if arr[0].RequiresGrad then
    arr[0].Grad := arr[0].Grad + CreateTensor(arr[0].Data.Shape, ADy.Val[0]);
end;

operator := (Val: double)V: TVariable;
begin
  V := TVariable.Create(Val);
  V.RequiresGrad := False;
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

end.
