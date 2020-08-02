{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit provides a native implementation in case of the absence of BLAS or
 any other accelerators.
}

unit noe.backend.native;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, noe;

{ Naive O(n^3) matmul implementation }
function MatMul_Native(A, B: TTensor): TTensor;

function MeanCol_Native(A: TTensor): TTensor;
function MeanRow_Native(A: TTensor): TTensor;
function SumCol_Native(A: TTensor): TTensor;
function SumRow_Native(A: TTensor): TTensor;

implementation

function MatMul_Native(A, B: TTensor): TTensor;
var
  i, j, k: longint;
  sum: double;
begin
  SetLength(Result.Val, A.Shape[0] * B.Shape[1]);
  for i := 0 to A.shape[0] - 1 do
    for j := 0 to B.Shape[1] - 1 do
    begin
      sum := 0;
      for k := 0 to A.Shape[1] - 1 do
        sum := sum + A.Val[i * A.Shape[1] + k] * B.Val[k * B.Shape[1] + j];
      Result.Val[i * B.Shape[1] + j] := sum;
    end;

  Result.ReshapeInplace([A.Shape[0], B.Shape[1]]);
end;

function MeanCol_Native(A: TTensor): TTensor;
begin
  Result := SumCol_Native(A) / A.Shape[0];
end;

function MeanRow_Native(A: TTensor): TTensor;
begin
  Result := SumRow_Native(A) / A.Shape[1];
end;

function SumCol_Native(A: TTensor): TTensor;
var
  i, j: longint;
begin
  //Result := TTensor.Create;
  SetLength(Result.Val, A.Shape[1]);
  Result.ReshapeInplace([1, A.Shape[1]]);
  for i := 0 to A.Shape[1] - 1 do
  begin
    Result.val[i] := 0;
    for j := 0 to A.Shape[0] - 1 do
      Result.val[i] := Result.val[i] + A.val[i + A.Shape[1] * j];
  end;
end;

function SumRow_Native(A: TTensor): TTensor;
var
  i, j: longint;
begin
  Result := SumCol_Native(A);
  SetLength(Result.Val, A.Shape[0]);
  Result.ReshapeInplace([A.Shape[0], 1]);
  for i := 0 to A.Shape[0] - 1 do
  begin
    Result.val[i] := 0;
    for j := 0 to A.Shape[1] - 1 do
      Result.val[i] := Result.val[i] + A.val[i * A.Shape[1] + j];
  end;
end;


end.

