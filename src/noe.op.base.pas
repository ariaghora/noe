{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit contains required basic operations for automatic gradient
 computation.
}

unit noe.op.base;

{$mode objfpc}

interface

uses
  Classes, SysUtils, noe.core, noe.Math, noe.autograd;

function Add(A, B: TVariable): TVariable;
function Subtract(A, B: TVariable): TVariable;
function Multiply(A, B: TVariable): TVariable;
function MultiplyC(A: TVariable; x: double): TVariable;
function MatMul(A, B: TVariable): TVariable;

function Sqr(A: TVariable): TVariable;
function Sqrt(A: TVariable): TVariable;
function ReLU(A: TVariable): TVariable;
function SumElement(A: TVariable): TVariable;

procedure BwAdd(arr: TVariableArr; ADy: TTensor);
procedure BwSubtract(arr: TVariableArr; ADy: TTensor);
procedure BwMultiply(arr: TVariableArr; ADy: TTensor);
procedure BwMultiplyC(arr: TVariableArr; ADy: TTensor);
procedure BwMatmul(arr: TVariableArr; ADy: TTensor);

procedure BwSqr(arr: TVariableArr; ADy: TTensor);
procedure BwSqrt(arr: TVariableArr; ADy: TTensor);
procedure BwReLU(arr: TVariableArr; ADy: TTensor);
procedure BwSumElement(arr: TVariableArr; ADy: TTensor);

operator +(A, B: TVariable) C: TVariable;
operator -(A, B: TVariable) C: TVariable;


implementation

function Add(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data + B.Data, 'Add', @BwAdd);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := A;
  Result.Prev[1] := B;
end;

function Subtract(A, B: TVariable): TVariable;
begin
  Result := TVariable.Create(A.Data + B.Data, 'Subtract', @BwSubtract);
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

function SumElement(A: TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.sum(A.Data), 'Sum', @BwSumElement);
  Result.RequiresGrad := True;

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := A;
end;

procedure BwAdd(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + ADy;
  arr[1].Grad := arr[1].Grad + ADy;
end;

procedure BwSubtract(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + ADy;
  arr[1].Grad := arr[1].Grad - ADy;
end;

procedure BwMultiply(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + noe.Math.Multiply(ADy, arr[1].Data);
  arr[1].Grad := arr[1].Grad + noe.Math.Multiply(ADy, arr[0].Data);
end;

procedure BwMultiplyC(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + noe.Math.Multiply(ADy, arr[1].Data);
end;

procedure BwMatmul(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + noe.Math.MatMul(ADy, arr[1].Data.T);
  arr[1].Grad := arr[1].Grad + noe.Math.MatMul(arr[0].Data.T, ADy);
end;

procedure BwSqr(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + (ADy * 2 * arr[0].Data);
end;

procedure BwMeanElement(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + (ADy * Ones(arr[0].Data.Shape));
end;

procedure BwSqrt(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + (ADy * 0.5 * 1 / (arr[0].Data ** 0.5));
end;

procedure BwReLU(arr: TVariableArr; ADy: TTensor);
var
  i: longint;
begin
  for i := 0 to Length(arr[0].Data.Val) - 1 do
    if arr[0].Data.Val[i] > 0 then
      arr[0].Grad.Val[i] := arr[0].Grad.Val[i] + ADy.Val[i];
end;

procedure BwSumElement(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + (ADy * Ones(arr[0].Data.Shape));
end;

operator +(A, B: TVariable)C: TVariable;
begin
  C := Add(A, B);
end;

operator -(A, B: TVariable)C: TVariable;
begin
  C := Subtract(A, B);
end;

end.
