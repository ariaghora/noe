unit noe.op.base;

{$mode objfpc}

interface

uses
  Classes, SysUtils, noe.core, noe.autograd;

var
  _node_count: longint;

type

  { TOPAdd }
  TOPAdd = class(TNode)
    A, B: TNode;
  public
    constructor Create(var _A, _B: TNode);
    function Eval: TTensor; overload;
    procedure Backward; override;
  end;

operator +(A, B: TNode) C: TOPAdd;

implementation

constructor TOPAdd.Create(var _A, _B: TNode);
begin
  inherited Create('Add');
  PrevNodes := PNodeList.Create;
  self.A := _A;
  self.B := _B;
  FShape := _A.Shape;
  PrevNodes.Add(@A);
  PrevNodes.Add(@B);

  self.Grad := FullTensor([self.Shape[0], self.Shape[1]], 0);
  self.PrevGrad := self.Grad;
end;

function TOPAdd.Eval: TTensor;
begin
  // TODO replace with Sum(PNodeList)
  Result := PrevNodes[0]^.Eval + PrevNodes[1]^.Eval;
end;

procedure TOPAdd.Backward;
begin
  PrevNodes[0]^.Grad := PrevNodes[0]^.Grad + self.PrevGrad;
  PrevNodes[1]^.Grad := PrevNodes[1]^.Grad + self.PrevGrad;
end;

operator +(A, B: TNode) C: TOPAdd;
begin
  C := TOPAdd.Create(A, B);
end;


end.
