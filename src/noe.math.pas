{ Extending FPC's math unit (and partially system unit) to be able to work
  with tensors }

unit noe.Math;

{$mode objfpc}{$H+}

interface


uses
  Classes, Math, noe.core;

type
  { Wrapping FPC's f:R->R unary functions in math unit }
  TUFunc = function(v: float): float;

  { Wrapping FPC's f:RxR->R binary functions in math unit }
  TBFunc = function(v1, v2: float): float;

function Add(A, B: TTensor): TTensor;
function Subtract(A, B: TTensor): TTensor;
function Multiply(A, B: TTensor): TTensor;

function Sum(M: TTensor): TTensor; overload;
function Sum(M: TTensor; axis: byte): TTensor; overload;

{ Helper to apply a function on each tensor's element }
function ApplyUfunc(A: TTensor; Func: TUFunc): TTensor;
function ApplyBfunc(A: TTensor; v: float; Func: TBFunc): TTensor;

{ Angle conversion }
function DegToRad(A: TTensor): TTensor; inline;
function RadToDeg(A: TTensor): TTensor; inline;

{ Logarithm functions }
function Log10(A: TTensor): TTensor;
function Log2(A: TTensor): TTensor;

operator ** (A: TTensor; expo: float) B: TTensor; inline;

{ Trigonometric functions }

{ Some of functions belong to system unit are in different format. Hence, they
  need to be wrapped to make them compatible. They are given suffix "F"
  (indicating float-valued function) to avoid confusion. }
function SinF(x: float): float;
function CosF(x: float): float;

function Sin(A: TTensor): TTensor;
function Cos(A: TTensor): TTensor;
function Tan(A: TTensor): TTensor;

{ Exponential functions }
function Power(A: TTensor; exponent: float): TTensor;

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

function Subtract(A, B: TTensor): TTensor;
var
  i: longint;
begin
  Assert(Length(A.Val) = Length(B.Val), MSG_ASSERTION_DIM_MISMATCH);

  Result := TTensor.Create;
  Result.Reshape(A.Shape);

  SetLength(Result.Val, Length(A.Val));
  for i := 0 to Length(A.Val) - 1 do
    Result.Val[i] := A.Val[i] - B.Val[i];
end;

function Multiply(A, B: TTensor): TTensor;
var
  i: longint;
begin
  Assert(Length(A.Val) = Length(B.Val), MSG_ASSERTION_DIM_MISMATCH);

  Result := TTensor.Create;
  Result.Reshape(A.Shape);

  SetLength(Result.Val, Length(A.Val));
  for i := 0 to Length(A.Val) - 1 do
    Result.Val[i] := A.Val[i] * B.Val[i];
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

operator ** (A: TTensor; expo: float)B: TTensor;
begin
  B := Power(A, expo);
end;

function SinF(x: float): float;
begin
  Result := System.Sin(x);
end;

function CosF(x: float): float;
begin
  Result := System.Cos(x);
end;

function Sin(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @SinF);
end;

function Cos(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @CosF);
end;

function Tan(A: TTensor): TTensor;
begin
  Result := ApplyUfunc(A, @Math.tan);
end;

function Power(A: TTensor; exponent: float): TTensor;
begin
  Result := ApplyBfunc(A, exponent, @Math.power);
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

function ApplyBfunc(A: TTensor; v: float; Func: TBFunc): TTensor;
var
  i: longint;
begin
  Result := TTensor.Create;
  Result.Reshape(A.Shape);
  SetLength(Result.val, Length(A.val));
  for i := 0 to length(A.val) - 1 do
    Result.val[i] := func(A.val[i], v);
end;

end.

