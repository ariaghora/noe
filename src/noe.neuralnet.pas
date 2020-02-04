unit noe.neuralnet;

{$mode objfpc}{$H+}

interface

uses
  Classes, fgl, noe, noe.Math, noe.utils, SysUtils;

type
  TLayer = class;
  TModel = class;

  TVariableList = specialize TFPGList<TVariable>;
  TLayerList    = specialize TFPGList<TLayer>;

  TActivationTypes = (atSigmoid, atReLU, atTanh, atNone);

  TDenseLayer   = class;
  TDropoutLayer = class;
  TSoftMaxLayer = class;

  { TLayer }

  TLayer = class
  private
    Params: TVariableArr;
  public
    function Eval(X: TVariable): TVariable; virtual; abstract;
    function GetParams: TVariableArr;
  end;

  { TDenseLayer }

  TDenseLayer = class(TLayer)
  private
    FActivation: TActivationTypes;
  public
    constructor Create(InSize, OutSize: longint; AActivation: TActivationTypes);
    function Eval(X: TVariable): TVariable; override;
    property Activation: TActivationTypes read FActivation write FActivation;
  end;

  { TDropoutLayer }

  TDropoutLayer = class(TLayer)
  private
    FDropoutRate: double;
    FUseDropout:  boolean;
    function GetUseDropout: boolean;
  public
    constructor Create(ADropoutRate: double);
    function Eval(X: TVariable): TVariable; override;
    property DropoutRate: double read FDropoutRate write FDropoutRate;
    property UseDropout: boolean read GetUseDropout write FUseDropout;
  end;

  { TSoftMaxLayer }

  TSoftMaxLayer = class(TLayer)
  private
    FAxis: longint;
  public
    constructor Create(AAxis: longint);
    function Eval(X: TVariable): TVariable; override;
  end;

  { TModel }

  TModel = class
    LayerList: TLayerList;
    Params:    TVariableArr;
  public
    constructor Create;
    constructor Create(Layers: array of TLayer); overload;
    function Eval(X: TVariable): TVariable;
    procedure AddLayer(Layer: TLayer);
  end;

{ Loss functions }
function AccuracyScore(predicted, actual: TTensor): double;
function CrossEntropyLoss(ypred, ytrue: TVariable): TVariable;
function L2Regularization(Model: TModel; Lambda: double = 0.001): TVariable;


implementation

function CrossEntropyLoss(ypred, ytrue: TVariable): TVariable;
begin
  Assert(ypred.Size = ytrue.Size, MSG_ASSERTION_DIFFERENT_LENGTH);
  Result := -Sum(ytrue * Log(ypred)) / ypred.Shape[0];
end;

function L2Regularization(Model: TModel; Lambda: double): TVariable;
var
  param: TVariable;
begin
  Result := 0;
  for param in Model.Params do
    Result := Result + Sum(param * param);
  Result   := Lambda * Result;
end;

function AccuracyScore(predicted, actual: TTensor): double;
var
  i: integer;
  tot: double;
begin
  tot := 0;
  for i := 0 to predicted.Size - 1 do
    { check if the sample is correctly classified (i.e., predicted = actual) }
    if predicted.GetAt(i) = actual.GetAt(i) then
      tot := tot + 1;
  Result  := tot / predicted.Size;
end;

{ TDropoutLayer }

function TDropoutLayer.GetUseDropout: boolean;
begin
  if GLOBAL_SKIP_GRAD then
    exit(False)
  else
    Result := self.FUseDropout;
end;

constructor TDropoutLayer.Create(ADropoutRate: double);
begin
  self.DropoutRate := ADropoutRate;
  self.UseDropout  := True;
end;

function TDropoutLayer.Eval(X: TVariable): TVariable;
var
  T: TTensor;
begin
  if Self.UseDropout then
  begin
    { FIXME: it works, but seems slow because of copy. Later the dropout can be
    applied directly on X data (i.e., pass by ref) }
    T      := X.Data;
    Result := X;
    Result.Data := T * RandomTensorBinomial(X.Shape, 1 - self.DropoutRate) *
      (1 / (1 - self.DropoutRate));
  end
  else
    Result := X;

end;

{ TSoftMaxLayer }

constructor TSoftMaxLayer.Create(AAxis: longint);
begin
  self.FAxis := AAxis;
end;

function TSoftMaxLayer.Eval(X: TVariable): TVariable;
begin
  Result := SoftMax(X, self.FAxis);
end;

{ TDenseLayer }

constructor TDenseLayer.Create(InSize, OutSize: longint; AActivation: TActivationTypes);

var
  W, b: TVariable;
begin
  inherited Create;
  Self.Activation := AActivation;

  { Xavier weight initialization }
  W := TVariable.Create(RandomTensorNormal([InSize, OutSize]) * 1 / (InSize ** 0.5));
  b := TVariable.Create(CreateTensor([1, OutSize], 0));
  SetRequiresGrad([W, b], True);

  SetLength(self.Params, 2);
  self.Params[0] := W;
  self.Params[1] := b;
end;

function TDenseLayer.Eval(X: TVariable): TVariable;
begin
  Result := X.Dot(self.Params[0]) + self.Params[1];

  case self.Activation of
    atReLU: Result := ReLU(Result);
    atTanh: Result := Tanh(Result);
    atSigmoid: raise ENotImplemented.Create(
        'Activation is not implemented yet.');
  end;
end;

{ TModel }

constructor TModel.Create;
begin
  self.LayerList := TLayerList.Create;
end;

constructor TModel.Create(Layers: array of TLayer);
var
  Layer: TLayer;
begin
  self.Create;
  for Layer in Layers do
    self.AddLayer(Layer);
end;

function TModel.Eval(X: TVariable): TVariable;
var
  Layer: TLayer;
begin
  Result := X;
  for Layer in self.LayerList do
    Result := Layer.Eval(Result);
end;

procedure TModel.AddLayer(Layer: TLayer);
var
  Param: TVariable;
begin
  self.LayerList.Add(Layer);
  for Param in Layer.Params do
  begin
    SetLength(self.Params, Length(self.Params) + 1);
    self.Params[Length(self.Params) - 1] := Param;
  end;
end;

{ TLayer }

function TLayer.GetParams: TVariableArr;
begin
  Result := self.Params;
end;

end.
