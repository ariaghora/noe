unit noe.op.base;

{$mode objfpc}

interface

uses
  Classes, SysUtils, noe.core, noe.autograd;

function VarOpAdd(arr: array of TVariable): TVariable;

procedure BackwardAdd(arr:TVariableArr; ADy: TTensor);

//operator +(A, B: TNode) C: TOPAdd;

implementation

function VarOpAdd(arr: array of TVariable): TVariable;
begin
  Result := TVariable.Create(arr[0].Data + arr[1].Data, 'Add', @BackwardAdd);

  SetLength(Result.FPrev, 2);
  Result.Prev[0] := arr[0];
  Result.Prev[1] := arr[1];
end;

procedure BackwardAdd(arr: TVariableArr; ADy: TTensor);
begin
  arr[0].Grad := arr[0].Grad + ADy;
  arr[1].Grad := arr[1].Grad + ADy;
end;

//operator +(A, B: TNode)C: TOPAdd;
//begin
//  C := OpAdd(A, B);
//end;




end.
