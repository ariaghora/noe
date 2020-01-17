{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit extends FPC's math unit (and partially system unit) to be able to
 work with tensors.

 To do:
  - implement matrix multiplication backend(s)
  - adapt more math functions from math.pas
  - implement iterate() that accepts callback to iterate over dimensions
}

unit noe.Math;

{$mode objfpc}{$H+}

interface


uses
  Classes, SysUtils, Math, RegExpr, fgl, noe.core, noe.utils, noe.backend.blas;

type
  { Wrapping FPC's f:R->R unary functions in math unit }
  TUFunc = function(v: float): float;

  { Wrapping FPC's f:RxR->R binary functions in math unit }
  TBFunc = function(v1, v2: float): float;

function Add(A, B: TTensor): TTensor;
function Subtract(A, B: TTensor): TTensor;
function Multiply(A, B: TTensor): TTensor;
function MatMul(A, B: TTensor): TTensor;

function Sum(M: TTensor): TTensor; overload;
function Sum(M: TTensor; axis: byte): TTensor; overload;

{ Evaluates the Einstein summation convention on the operands.
  The initial implementation is heavily inspired from Kyle Hundman's attempt to
  mirror numpy's einsum, so not all operations are supported. Any helps are
  welcome. }
function Einsum(Subscripts: string; Pots: array of TTensor): TTensor;

{ Helper to apply a function on each tensor's element }
function ApplyUfunc(A: TTensor; Func: TUFunc): TTensor;
function ApplyBfunc(A: TTensor; v: float; Func: TBFunc): TTensor;

{ Angle conversion }
function DegToRad(A: TTensor): TTensor; inline;
function RadToDeg(A: TTensor): TTensor; inline;

{ Logarithm functions }
function Log10(A: TTensor): TTensor;
function Log2(A: TTensor): TTensor;

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

operator ** (A: TTensor; expo: float) B: TTensor; inline;

function Transpose(T: TTensor; dims: array of longint): TTensor;

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

function MatMul(A, B: TTensor): TTensor;
begin
  Assert((length(A.Shape) <= 2) and (length(B.Shape) <= 2),
    'Tensor dimension must be <= 2.');

  { calculates matrix multiplication according to the backend }
  if noe.core.NoeConfig.useBLAS then
    Result := MatMul_BLAS(A, B)
  else
    Result := FullTensor([2, 2], 0);
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

function Transpose(T: TTensor; dims: array of longint): TTensor;
var
  resultedIdx, dimsLetter: string;
  i: longint;
begin
  dimsLetter := DimsToLetter(dims);
  Assert(Length(dims) = length(T.Shape), 'dims length does not match tensor dimension');
  resultedIdx := DimsToLetter(dims);
  for i := 0 to Length(dims) - 1 do
    resultedIdx[i + 1] := dimsLetter.Chars[dims[i]];
  Result := Einsum(dimsLetter + '->' + resultedIdx, [T]);
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

function Einsum(Subscripts: string; Pots: array of TTensor): TTensor;
type
  TNameDimsMap = specialize TFPGMap<string, TIntVector>;
  TStringIntMap = specialize TFPGMap<string, longint>;
  TIntVectorList = specialize TFPGList<TIntVector>;
  TIntVectorArr = array of TIntVector;
var
  re: TRegExpr;
  match, keepGoing, skipCombo, found: boolean;
  TmpTensor, output: TTensor;
  i, j, h, len, dim: longint;
  split, tables: TStringArray;
  broadcastList, flatTables, originalTables, uniqueTables: ansistring;
  nameAndDims: TNameDimsMap;
  uniqueDict: TStringIntMap;
  comb, bcomb, indices, flatDims, broadcastDims, combinations: TIntVector;
  forMultiPlying: TFloatVector;
  dims, combos, broadcastCombos: TIntVectorArr;
  s: string;
  plug, Value, v: float;

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
    re := TRegExpr.Create('(.)\1');
    match := re.Exec(Subscripts);
    { there are repeated letters, return diagonal }
    if match then
    begin
      Assert(Pots[0].Shape[0] = Pots[0].Shape[1], 'Cannot collapse index ' +
        re.Match[0].Chars[0]);

      Result := TTensor.Create;
      len := Pots[0].Shape[0];
      SetLength(Result.Val, len);
      Result.Reshape([len]);
      for i := 0 to len - 1 do
        Result.Val[i] := Pots[0].GetAt([i, i]).Val[0];
    end

    { tensor dot multiplication and specific dimension broadcasting }
    else
    begin
      split := Subscripts.Split('->');
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
      flatTables := '';
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

      Result := FullTensor(broadcastDims, 0.0);

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
              begin
                indices[j] := comb[uniqueTables.IndexOf(tables[i].Chars[j])];
              end;

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
  begin
    Result := Math.sum(Einsum(Subscripts + '->', Pots).Val);
  end;

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
