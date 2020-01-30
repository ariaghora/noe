unit noe.autograd;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, noe.core;

var
  GLOBAL_NODE_COUNT: integer;

type

  { TVariable }

  PVariable = ^TVariable;

  TVariable     = class;
  TVariableArr  = array of TVariable;
  PVariableArr  = array of ^TVariable;
  TBackwardFunc = procedure(arr: TVariableArr; ADy: TTensor);

  TVariable = class
    FTensor: TTensor;
    FTensorPtr: PTensor;
    FGrad:   TTensor;
    FID:     longint;
    FIsLeaf: boolean;
    FRequiresGrad: boolean;
    FPrev:   TVariableArr;
  private
    FBackwardFunc: TBackwardFunc;
    FName: string;
    procedure SetData(AValue: TTensor);
  public
    constructor Create; overload;
    constructor Create(AName: string); overload;
    constructor Create(ATensor: TTensor); overload;
    constructor Create(ATensor: TTensor; AName: string); overload;
    constructor Create(ATensor: TTensor; AName: string;
      ABackwardFunc: TBackwardFunc); overload;
    constructor Create(ATensor: TTensor; AName: string;
      ABackwardFunc: TBackwardFunc; AIsLeaf: boolean); overload;
    procedure Backpropagate;
    procedure Step(LearningRate: double);
    procedure ZeroGrad;
    property BackwardFunc: TBackwardFunc read FBackwardFunc write FBackwardFunc;
    property Data: TTensor read FTensor write SetData;
    property Grad: TTensor read FGrad write FGrad;
    property ID: longint read FID write FID;
    property IsLeaf: boolean read FIsLeaf write FIsLeaf;
    property Name: string read FName write FName;
    property Prev: TVariableArr read FPrev write FPrev;
    property RequiresGrad: boolean read FRequiresGrad write FRequiresGrad;
    property Tensor: TTensor read FTensor write FTensor;

    { Math helpers }
    function MatMul(Other: TVariable): TVariable;
  end;

function TopologicalSort(T: TVariable): TVariableArr;
procedure BackwardGraph(const T: TVariable);

operator in (T: TVariable; arr: array of TVariable) b: boolean;

implementation

uses
  noe.op.base;

function TopologicalSort(T: TVariable): TVariableArr;
var
  Seen, Sorted: TVariableArr;
  prv: TVariable;

  procedure TopoHelper(v: TVariable);
  begin
    if (not(v in Seen)) and (not(v.IsLeaf)) then
    begin
      SetLength(Seen, Length(seen) + 1);
      Seen[Length(Seen) - 1] := v;
      for prv in v.Prev do
        TopoHelper(prv);

      if v.RequiresGrad then
      begin
        SetLength(Sorted, Length(Sorted) + 1);
        Sorted[Length(Sorted) - 1] := v;
      end;
    end;
  end;
begin
  TopoHelper(T);
  Result := Sorted;
end;

procedure BackwardGraph(const T: TVariable);
var
  Sorted: TVariableArr;
  i:longint;
begin
  Sorted := TopologicalSort(T);

  T.Grad := Ones(T.Data.Shape);
  for i := length(Sorted) - 1 downto 0 do
    Sorted[i].BackwardFunc(Sorted[i].Prev, Sorted[i].FGrad);
end;

operator in(T: TVariable; arr: array of TVariable)b: boolean;
var
  Tmp:TVariable;
begin
  result := false;
  for Tmp in arr do
    if T.ID = Tmp.ID then
    begin
      result := true;
      exit;
    end;
end;

procedure TVariable.SetData(AValue: TTensor);
begin
  if FTensor = AValue then
    Exit;
  FTensor := AValue;
end;

constructor TVariable.Create;
begin
  self.Create(nil, '', nil, True);
end;

constructor TVariable.Create(AName: string);
begin
  self.Create(nil, AName, nil, True);
end;

constructor TVariable.Create(ATensor: TTensor);
begin
  self.Create(ATensor, '', nil, True);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string);
begin
  self.Create(ATensor, AName, nil, True);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string;
  ABackwardFunc: TBackwardFunc);
begin
  { it has a Backpropagate function, so it must be non-leaf }
  self.Create(ATensor, AName, ABackwardFunc, False);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string;
  ABackwardFunc: TBackwardFunc; AIsLeaf: boolean);
begin
  self.Data := ATensor;
  self.Name := AName;
  self.BackwardFunc := ABackwardFunc;
  self.IsLeaf := AIsLeaf;

  { always true on creation unless specified otherwise }
  self.RequiresGrad:= False;

  self.ZeroGrad;

  self.FID := GLOBAL_NODE_COUNT;
  Inc(GLOBAL_NODE_COUNT);
end;

procedure TVariable.Backpropagate;
begin
  BackwardGraph(self);
end;

procedure TVariable.Step(LearningRate: double);
begin
  if Self.RequiresGrad then
    self.Data := self.Data - LearningRate * self.Grad;
end;

procedure TVariable.ZeroGrad;
begin
  self.Grad := Zeros(self.Tensor.Shape);
end;

function TVariable.MatMul(Other: TVariable): TVariable;
begin
  Result := noe.op.base.MatMul(self, Other);
end;


initialization
  GLOBAL_NODE_COUNT := 0;
end.
