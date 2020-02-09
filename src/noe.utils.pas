{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit implement some helper functionalities, such as some operator
 overloadings, which I think will be helpful.
}

unit noe.utils;

{$mode objfpc}{$H+}

interface

uses
  SysUtils,
  strutils,
  fgl,
  Classes,
  math,
  noe;

type
  TIntVector = array of longint;
  TDoubleList = specialize TFPGList<double>;
  TDoubleIntMap = specialize TFPGMap<double, longint>;

  { One-hot encode categorical labels }
  TOneHotEncoder = class
    unique: TDoubleList;
    function Encode(T: TTensor): TTensor;
    function Decode(T: TTensor): TTensor;
  private
    LabelToIndexMap: TDoubleIntMap;
  end;

function IntVectorEquals(v1, v2: TIntVector): boolean;
function ReverseIntArr(A: array of longint): TIntVector;
function ReverseFloatArr(A: array of Double): TFloatVector;

{ Sorting chars in a string using bubble sort. Not for big strings. }
function SortStr(s: string; ascending: boolean = True): string; inline;
function StandardScaler(X:TTensor): TTensor;

procedure NoeLog(tag, msg: string);
procedure VisualizeMatrix(T: TTensor);

operator in (substr, mainstr: string) b: boolean;
operator in (str: string; arr: array of string) b: boolean;
operator in (x: double; arr: array of double) b: boolean;
operator = (a, b: array of longint) c: boolean;

implementation

uses
  noe.Math;


function IntVectorEquals(v1, v2: TIntVector): boolean;
var
  i: longint;
begin
  Result := True;
  if not (length(v1) = length(v2)) then Exit(False);
  for i := 0 to length(v1) - 1 do
    if v1[i] <> v2[i] then
    begin
      Exit(False);
    end;
end;

function ReverseIntArr(A: array of longint): TIntVector;
var
  i: longint;
begin
  SetLength(Result, Length(A));
  for i := Length(A) - 1 downto 0 do
    Result[Length(A) - i - 1] := A[i];
end;

function ReverseFloatArr(A: array of Double): TFloatVector;
var
  i: longint;
begin
  SetLength(Result, Length(A));
  for i := Length(A) - 1 downto 0 do
    Result[Length(A) - i - 1] := A[i];
end;

function SortStr(s: string; ascending: boolean = True): string;
var
  i, j: integer;
  tmp: char;
  tmpstr: string;
  compSatisfied: boolean;
begin
  tmpstr := s;
  for i := 1 to Length(s) do
  begin
    for j := 1 to Length(s) do
    begin
      if ascending then
        compSatisfied := tmpstr[i] < tmpstr[j]
      else
        compSatisfied := tmpstr[i] > tmpstr[j];

      if compSatisfied then
      begin
        tmp := tmpstr[i];
        tmpstr[i] := tmpstr[j];
        tmpstr[j] := tmp;
      end;
    end;
  end;
  Result := tmpstr;
end;

function CompareDouble(const x, y: double): integer;
begin
  if x = y then
    Result := 0
  else if x < y then
    Result := -1
  else
    Result := 1;
end;

function StandardScaler(X: TTensor): TTensor;
var
  mu, std: TTensor;
begin
  mu := Mean(X, 0);
  std := (Mean((X - mu) ** 2, 0)) ** 0.5;

  Result := ((X - mu)/std);
end;

procedure NoeLog(tag, msg: string);
begin
  if noe.NoeConfig.debug and IsConsole then
  begin
    WriteLn(tag + ': ' + msg);
  end;
end;

procedure VisualizeMatrix(T: TTensor);
var
  i, j: integer;
  maxval: double;
begin
  Assert(T.NDims = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  maxval := maxvalue(T.Val);
  for i := 0 to T.Shape[0] - 1 do
  begin
    for j := 0 to T.Shape[1] - 1 do
    begin
      if (T.GetAt(i, j) / maxval) > (4/5) then write(#178)
      else if (T.GetAt(i, j) / maxval) > (3/5) then write(#177)
      else if (T.GetAt(i, j) / maxval) > (2/5) then write(#176)
      else if (T.GetAt(i, j) / maxval) > (1/5) then write(#247)
      else write(' ');
    end;
    writeln;
    //s := s + sLineBreak;
  end;
  //WriteLn(s);
end;

operator in (substr, mainstr: string)b: boolean;
begin
  b := AnsiContainsStr(mainstr, substr);
end;

operator in(str: string; arr: array of string)b: boolean;
var
  i: longint;
begin
  result := false;
  for i:=0 to length(arr)-1 do
    if str = arr[i] then
    begin
      result := true;
      exit;
    end;
end;

operator in(x: double; arr: array of double)b: boolean;
var
  i: longint;
begin
  result := false;
  for i:=0 to length(arr)-1 do
    if x = arr[i] then
    begin
      result := true;
      exit;
    end;
end;

operator = (a, b: array of longint) c: boolean;
var
  i: longint;
begin
  Assert(length(a) = length(b), MSG_ASSERTION_DIFFERENT_LENGTH);
  c := True;
  for i := 0 to length(a) - 1 do
    if a[i] <> b[i] then
    begin
      c := False;
      exit;
    end;
end;

{ TOneHotEncoder }

function TOneHotEncoder.Encode(T: TTensor): TTensor;
var
  i: double;
  j, row: longint;
begin
  Assert(T.NDims = 1, MSG_ASSERTION_RANK_1_TENSORS_ONLY);

  { get unique labels }
  unique := TDoubleList.Create;
  for i in T.Val do
    if (unique.IndexOf(i) < 0) then
      unique.Add(i);
  unique.Sort(@CompareDouble);

  { Create zeros as the placeholder }
  Result := Zeros([T.Size, unique.Count]);

  LabelToIndexMap := TDoubleIntMap.Create;
  for j := 0 to unique.Count - 1 do
    LabelToIndexMap.Add(unique.Items[j], j);

  { Actual data handling }
  for row := 0 to Result.Shape[0] - 1 do
    Result.SetAt(row, LabelToIndexMap.KeyData[T.Val[row]], 1.0);
end;

function TOneHotEncoder.Decode(T: TTensor): TTensor;
var
  Indices: TTensor;
  i: longint;
begin
  Assert(T.NDims = 2, MSG_ASSERTION_RANK_2_TENSORS_ONLY);
  Indices := Squeeze(ArgMax(T, 1));

  Result.ReshapeInplace([Indices.Size]);
  SetLength(Result.Val, Indices.Size);
  for i := 0 to Indices.Size - 1 do
    Result.SetAt(i, unique[Round(Indices.GetAt(i))]);
end;


end.
