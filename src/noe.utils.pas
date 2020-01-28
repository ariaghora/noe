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
  sysutils, strutils, classes, noe.core;

type
  TIntVector = array of longint;
function ReverseIntArr(A: array of longint): TIntVector;

{ Sorting chars in a string using bubble sort. Not for big strings. }
function SortStr(s: string; ascending:boolean=true): string; inline;

{ Create a rank-2 tensor (a matrix) from a CSV file }
function ReadCSV(fileName:string):TTensor;

procedure NoeLog(tag, msg: string);

operator in (substr, mainstr: string) b: boolean;
operator in (str: string; arr: array of string) b: boolean;
operator = (a, b: array of longint) c: boolean;

implementation

function ReverseIntArr(A: array of longint): TIntVector;
var
  i: longint;
begin
  SetLength(Result, Length(A));
  for i := Length(A) - 1 downto 0 do
    Result[Length(A) - i - 1] := A[i];
end;

function SortStr(s: string; ascending:boolean=true): string;
var
  i,j:integer;
  tmp:char;
  tmpstr:string;
  compSatisfied:boolean;
begin
  tmpstr := s;
  for i:=1 to Length(s) do
  begin
    for j:=1 to Length(s) do
    begin
      if ascending then
         compSatisfied := tmpstr[i] < tmpstr[j]
      else
         compSatisfied := tmpstr[i] > tmpstr[j];

      if compSatisfied then
      begin
        tmp := tmpstr[i];
        tmpstr[i]:=tmpstr[j];
        tmpstr[j]:=tmp;
      end;
    end;
  end;
  result:=tmpstr;
end;

function ReadCSV(fileName: string): TTensor;
var
  s, number: string;
  sl: TStringList;
  InFile: text;
  RowCount, ColCount, offset: longint;
begin
  Assign(InFile, fileName);
  Reset(InFile);

  sl := TStringList.Create;
  sl.StrictDelimiter := True;

  { first run: estimate the RowCount & ColCount }
  ReadLn(InFile, s);
  sl.CommaText := s;
  ColCount := sl.Count;

  RowCount := 1;
  while not eof(InFile) do
  begin
    Inc(RowCount);
    ReadLn(InFile);
  end;

  { actual data handle }
  Result := TTensor.Create;
  Result.Reshape([RowCount, ColCount]);
  SetLength(Result.Val, RowCount * ColCount);

  offset := 0;
  Reset(InFile);
  while not eof(InFile) do
  begin
    ReadLn(InFile, s);
    sl.CommaText := s;

    for number in sl do
    begin
      Result.Val[offset] := StrToFloat(number);
      Inc(offset);
    end;
  end;

  Close(InFile);
  sl.Free;
end;

procedure NoeLog(tag, msg: string);
begin
  if noe.core.NoeConfig.debug and IsConsole then
  begin
    WriteLn(tag + ': ' + msg);
  end;
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

operator = (a, b: array of longint) c: boolean;
var
  i: longint;
begin
  Assert(length(a) = length(b), MSG_ASSERTION_DIFFERENT_LENGTH);
  c := True;
  for i := 0 to length(a) do
    if a[i] <> b[i] then
    begin
      c := False;
      exit;
    end;
end;

end.

