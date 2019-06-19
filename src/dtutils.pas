{ A unit providing some useful helper functions }
unit DTUtils;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GQueue, DTCore;

type
  TItemRadixSortf = double;

function Getunique(y: TFloatVector): TFloatVector;

implementation

procedure RadixSortf(var a: array of TItemRadixSortf);
const
  BASE = 16;
type
  TQueueIRS = specialize TQueue< TItemRadixSortf >;
var
  jono: array[0 .. BASE - 1] of TQueueIRS;
  max: TItemRadixSortf;
  i, k: integer;

  procedure pick;
  var
    i, j: integer;
  begin
    i := 0;
    j := 0;
    while i < high(a) do
    begin
      while not jono[j].IsEmpty do
      begin
        a[i] := jono[j].Front;
        jono[j].Pop;
        Inc(i);
      end;
      Inc(j);
    end;
  end;

begin
  max := high(a);
  for i := 0 to BASE - 1 do
    jono[i] := TQueueIRS.Create;
  for i := low(a) to high(a) do
  begin
    if a[i] > max then
      max := a[i];
    jono[abs(round(a[i]) mod BASE)].Push(a[i]);
  end;
  pick;
  k := BASE;
  while max > k do
  begin
    for i := low(a) to high(a) do
      jono[abs(round(a[i]) div k mod BASE)].Push(a[i]);
    pick;
    k := k * BASE;
  end;
  for i := 0 to BASE - 1 do
    jono[i].Free;

end;

function Getunique(y: TFloatVector): TFloatVector;
var
  ySorted, unique: TFloatVector;
  i, j, c, cntUnique: integer;
  currFound: real;
begin
  {
    perform sorting on the (copied) 'y' to reduce the time complexity to
    O(log n) for finding unique class labels.
  }
  SetLength(ySorted, Length(y));
  ySorted := copy(y, 0, MaxInt);
  radixsortf(ySorted);

  cntUnique := 1;
  SetLength(unique, cntUnique);
  currFound := ySorted[0];
  unique[0] := currFound;

  for i := 0 to Length(y) - 1 do
  begin
    if (currFound <> ySorted[i]) then
    begin
      Inc(cntUnique);
      currFound := ySorted[i];
      SetLength(unique, cntUnique);
      unique[cntUnique - 1] := currFound;
    end;
  end;
  Result := unique;
end;

end.

