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
  strutils;

{ Sorting chars in a string using bubble sort. Not for big strings. }
function SortStr(s: string; ascending:boolean=true): string; inline;

procedure NoeLog(tag, msg: string);

operator in (substr, mainstr: string) b: boolean;
operator in (str: string; arr: array of string) b: boolean;
operator = (a, b: array of longint) c: boolean;

implementation

uses
  noe.core;

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

