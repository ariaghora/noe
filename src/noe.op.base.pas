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

function Add(arr: array of TVariable): TVariable;
function Subtract(arr: array of TVariable): TVariable;
function Multiply(arr: array of TVariable): TVariable;
function MatMul(arr: array of TVariable): TVariable;

function Sqr(arr: array of TVariable): TVariable;
function Sqrt(arr: array of TVariable): TVariable;
function SumElement(arr: array of TVariable): TVariable;

procedure BwAdd(arr: TVariableArr; ADy: TTensor);
procedure BwSubtract(arr: TVariableArr; ADy: TTensor);
procedure BwMultiply(arr: TVariableArr; ADy: TTensor);
procedure BwMatmul(arr: TVariableArr; ADy: TTensor);

procedure BwSqr(arr: TVariableArr; ADy: TTensor);
procedure BwSqrt(arr: TVariableArr; ADy: TTensor);
procedure BwSumElement(arr: TVariableArr; ADy: TTensor);

operator +(A, B: TVariable) C: TVariable;
operator -(A, B: TVariable) C: TVariable;


implementation

function Add(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(arr[0].Data + arr[1].Data, 'Add', @BwAdd);

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := arr[0];
  Result.Prev[1] := arr[1];
end;

function Subtract(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(arr[0].Data + arr[1].Data, 'Subtract', @BwSubtract);

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := arr[0];
  Result.Prev[1] := arr[1];
end;

function Multiply(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.Multiply(arr[0].Data, arr[1].Data),
    'Multiply', @BwMultiply);

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := arr[0];
  Result.Prev[1] := arr[1];
end;

function MatMul(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.MatMul(arr[0].Data, arr[1].Data),
    'MatMul', @BwMatmul);

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := arr[0];
  Result.Prev[1] := arr[1];
end;

function Sqr(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(arr[0].Data ** 2, 'Sqr', @BwSqr);

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := arr[0];
end;

function Sqrt(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(arr[0].Data ** 0.5, 'Sqrt', @BwSqrt);

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := arr[0];
end;

function SumElement(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(noe.Math.sum(arr[0].Data), 'Sum', @BwSumElement);

  SetLength(Result.FPrev, 1);
  Result.Prev[0] := arr[0];
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
  arr[0].Grad := arr[0].Grad + (ADy * 0.5 * 1/(arr[0].Data ** 0.5));
end;

procedure BwSumElement(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + (ADy * Ones(arr[0].Data.Shape));
end;

operator +(A, B: TVariable)C: TVariable;
begin
  C := Add([A, B]);
end;

operator-(A, B: TVariable)C: TVariable;
begin
  C := Subtract([A, B]);
end;

end.
