unit noe.autograd;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fgl, noe.core;

var
  GLOBAL_NODE_COUNT: integer;

type

  { Abstract data type as the backbone of the computational graph.

    All variables and operations are derived from this class.}
  PNode = ^TNode;
  PNodeList = specialize TFPGList<PNode>;

  { TNode }

  TNode = class
    FShape: TIntVector;
  private
    FGrad: TTensor;
    FName: string;
    FPrev: PNodeList;

    FMat: TTensor;
    FID: longint;
    FPrevGrad: TTensor;
  public
    constructor Create(AName: string);
    constructor Create(AMat: TTensor; AName: string);
    function Eval: TTensor;
    procedure Backward; virtual; abstract;
    procedure SetShape(ShapeVals: array of longint);
    property Mat: TTensor read FMat write FMat;
    property Grad: TTensor read FGrad write FGrad;
    property ID: longint read FID write FID;
    property PrevNodes: PNodeList read FPrev write FPrev;
    property PrevGrad: TTensor read FPrevGrad write FPrevGrad;
    property Shape: TIntVector read FShape;// write SetShape;
    property Name: string read FName write FName;
  end;

  { TVariable }

  TVariable = class(TNode)
  public
    constructor Create(var AMat: TTensor; AName: string);
    procedure backward; override;
    function Eval: TTensor;
  end;

procedure PrintMatrix(M: TNode); overload;

implementation

{ TNode }
constructor TNode.Create(AName: string);
begin
  self.Name := AName;
  self.ID := GLOBAL_NODE_COUNT;
  self.PrevNodes := PNodeList.Create;

  Inc(GLOBAL_NODE_COUNT);
end;

constructor TNode.Create(AMat: TTensor; AName: string);
begin
  self.Create(AName);
  Self.Mat := Mat;

  // TODO require_grad checking
  self.Mat := AMat;
  //self.FShape.Height := self.Mat.Shape.Height;
  //self.FShape.Width := self.Mat.Shape.Width;
  self.SetShape(self.Mat.Shape);
  //self.FShape[0] := self.Mat.Shape[0];
  //self.FShape[1] := self.Mat.Shape[1];

  self.FGrad := FullFloat(self.Shape[0], self.Shape[1], 0);
  self.PrevGrad := self.FGrad;
end;

function TNode.Eval: TTensor;
begin
  Result := self.Mat;
end;

procedure TNode.SetShape(ShapeVals: array of longint);
var
  i: longint;
begin
  SetLength(self.FShape, Length(ShapeVals));
  for i := 0 to Length(ShapeVals) - 1 do
    self.FShape[i] := ShapeVals[i];
end;

{ TVariable }
constructor TVariable.Create(var AMat: TTensor; AName: string);
begin
  inherited Create(AMat, AName);
  self.Mat := AMat;
  self.FShape[0] := self.Mat.Shape[0];
  self.FShape[1] := self.Mat.Shape[1];

end;

procedure TVariable.backward;
begin

end;

function TVariable.Eval: TTensor;
begin
  Result := self.Mat;
end;

procedure PrintMatrix(M: TNode); overload;
var
  preamble: string;
begin
  preamble := 'shape=' + IntToStr(M.Eval.Shape[0]) + 'x' + IntToStr(M.Eval.Shape[1]);


  preamble := preamble + ' (TNode), Name=' + M.Name;
  noe.core.PrintTensor(M.Eval, preamble);
end;

initialization
  GLOBAL_NODE_COUNT := 0;
end.
