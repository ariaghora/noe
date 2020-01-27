unit noe.autograd;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fgl, noe.core;

var
  GLOBAL_NODE_COUNT: integer;

type

  { TVariable }

  PVariable      = ^TVariable;
  PVariableArray = array of PVariable;

  TVariable     = class;
  TVariableArr  = array of TVariable;
  TBackwardFunc = procedure(arr: TVariableArr; ADy: TTensor);

  TVariable = class
    FTensor: TTensor;
    FGrad:   TTensor;
    FID:     integer;
    FIsLeaf: boolean;
    FPrev:   TVariableArr;

  private
    FBackwardFunc: TBackwardFunc;
    FName: string;
    procedure SetData(AValue: TTensor);
  public
    constructor Create; overload;
    constructor Create(AName: string); overload;
    constructor Create(ATensor: TTensor; AName: string); overload;
    constructor Create(ATensor: TTensor; AName: string;
      ABackwardFunc: TBackwardFunc); overload;
    procedure Backward;
    procedure Step(LearningRate: double);
    procedure ZeroGrad;
    property BackwardFunc: TBackwardFunc read FBackwardFunc;
    property Data: TTensor read FTensor write SetData;
    property Grad: TTensor read FGrad write FGrad;
    property IsLeaf: boolean read FIsLeaf write FIsLeaf;
    property Name: string read FName write FName;
    property Prev: TVariableArr read FPrev write FPrev;
    property Tensor: TTensor read FTensor write FTensor;
  end;

implementation

procedure TVariable.SetData(AValue: TTensor);
begin
  if FTensor = AValue then
    Exit;
  FTensor := AValue;
end;

constructor TVariable.Create;
begin
  self.FName := '';
  self.FIsLeaf := True;
  self.FID := GLOBAL_NODE_COUNT;

  Inc(GLOBAL_NODE_COUNT);
end;

constructor TVariable.Create(AName: string);
begin
  self.FName := AName;
end;

{ TVariable }
constructor TVariable.Create(ATensor: TTensor; AName: string);
begin
  self.Create(AName);
  self.FTensor := ATensor;
  self.ZeroGrad;
end;

constructor TVariable.Create(ATensor: TTensor; AName: string;
  ABackwardFunc: TBackwardFunc);
begin
  self.Create(ATensor, AName);
  self.FBackwardFunc := ABackwardFunc;
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
