unit noe.math;

{ Math interface }

{$mode objfpc}{$H+}

interface

uses
  Classes, noe.core;

function Add(A, B: TTensor): TTensor;
function Sum(M: TTensor): TTensor; overload;
function Sum(M: TTensor; axis: byte): TTensor; overload;

implementation

function Add(A, B: TTensor): TTensor;
var
  i: longint;
begin
  Assert(Length(A.Val) = Length(B.Val), MSG_ASSERTION_DIM_MISMATCH);

  Result := TTensor.Create;
  Result.Reshape(A.Shape);

  SetLength(Result.Val, Length(A.Val));
  for i := 0 to Length(A.Val) - 1 do
    Result.Val[i] := A.Val[i] + B.Val[i];
end;

function Sum(M: TTensor): TTensor;
var
  i: longint;
  tot: single;
begin
  tot := 0;
  for i := 0 to length(M.Val) - 1 do
    tot := tot + M.val[i];
  Result := tot;
end;

function Sum(M: TTensor; axis: byte): TTensor;
var
  i, j: longint;
begin
  Assert(axis <= 1, MSG_ASSERTION_INVALID_AXIS);
  if axis = 0 then
  begin
    SetLength(Result.Val, M.Shape[1]);
    Result.Reshape([1, M.Shape[1]]);
    for i := 0 to M.Shape[1] - 1 do
    begin
      Result.val[i] := 0;
      for j := 0 to M.Shape[0] - 1 do
        Result.val[i] := Result.val[i] + M.val[i + M.Shape[1] * j];
    end;
  end
  else
  begin
    SetLength(Result.Val, M.Shape[0]);
    Result.Reshape([M.Shape[0], 1]);
    for i := 0 to M.Shape[0] - 1 do
    begin
      Result.val[i] := 0;
      for j := 0 to M.Shape[1] - 1 do
        Result.val[i] := Result.val[i] + M.val[i * M.Shape[1] + j];
    end;
  end;
end;

end.

