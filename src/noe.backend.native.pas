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
  Classes, SysUtils, noe.core;

function MatMul_Native(A, B: TTensor): TTensor;

implementation

function MatMul_Native(A, B: TTensor): TTensor;
var
  i, j, k: longint;
  sum: double;
begin
  Result := TTensor.Create;
  SetLength(Result.Val, A.Shape[0] * B.Shape[1]);
  for i := 0 to A.shape[0] - 1 do
    for j := 0 to B.Shape[1] - 1 do
    begin
      sum := 0;
      for k := 0 to A.Shape[1] - 1 do
        sum := sum + A.Val[i * A.Shape[1] + k] * B.Val[k * B.Shape[1] + j];
      Result.Val[i * B.Shape[1] + j] := sum;
    end;

  Result.Reshape([A.Shape[0], B.Shape[1]]);
end;



end.

