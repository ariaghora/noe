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
    procedure Backward;
    procedure Step(LearningRate: double);
    procedure ZeroGrad;
    property BackwardFunc: TBackwardFunc read FBackwardFunc write FBackwardFunc;
    property Data: TTensor read FTensor write SetData;
    property Grad: TTensor read FGrad write FGrad;
    property ID: longint read FID write FID;
    property IsLeaf: boolean read FIsLeaf write FIsLeaf;
    property Name: string read FName write FName;
    property Prev: TVariableArr read FPrev write FPrev;
    property Tensor: TTensor read FTensor write FTensor;
  end;

function TopologicalSort(T: TVariable): TVariableArr;
procedure BackwardGraph(const T: TVariable);

operator in (T: TVariable; arr: array of TVariable) b: boolean;

implementation

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

      SetLength(Sorted, Length(Sorted) + 1);
      Sorted[Length(Sorted) - 1] := v;
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
  begin
    Sorted[i].Backward;
  end;
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
  { it has a backward function, so it must be non-leaf }
  self.Create(ATensor, AName, ABackwardFunc, False);
end;

constructor TVariable.Create(ATensor: TTensor; AName: string;
  ABackwardFunc: TBackwardFunc; AIsLeaf: boolean);
begin
  self.Data := ATensor;
  self.Name := AName;
  self.BackwardFunc := ABackwardFunc;
  self.IsLeaf := AIsLeaf;

  self.ZeroGrad;

  self.FID := GLOBAL_NODE_COUNT;
  Inc(GLOBAL_NODE_COUNT);
end;

procedure TVariable.Backward;
begin
  Self.BackwardFunc(self.Prev, self.FGrad);
end;

procedure TVariable.Step(LearningRate: double);
begin
  self.Data := self.Data - LearningRate * self.Grad;
end;

procedure TVariable.ZeroGrad;
begin
  self.Grad := Zeros(self.Tensor.Shape);
end;


initialization
  GLOBAL_NODE_COUNT := 0;
end.
