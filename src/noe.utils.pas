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

operator in (substr, mainstr: string) b: boolean;
operator = (a, b: array of longint) c: boolean;

implementation

uses
  noe.core;

operator in (substr, mainstr: string)b: boolean;
begin
  b := AnsiContainsStr(mainstr, substr);
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

