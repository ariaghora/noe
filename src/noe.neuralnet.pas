{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit contains the interface for high-level neural network API. Specifically,
 it contains the implementation of layers, optimizers, and loss functions.
}
unit noe.neuralnet;

{$mode objfpc}{$H+}

interface

uses
  Classes, fgl, fpjson, jsonparser, Math, noe, noe.Math, SysUtils;

type
  TLayer = class;
  TModel = class;

  TVariableList = specialize TFPGList<TVariable>;
  TLayerList    = specialize TFPGList<TLayer>;

  TBatchNormLayer = class;
  TConv2dLayer    = class;
  TDenseLayer     = class;
  TDropoutLayer   = class;
  TFlattenLayer   = class;
  TLeakyReLULayer = class;
  TReLULayer      = class;
  TSigmoidLayer   = class;
  TSoftMaxLayer   = class;
  TTanhLayer      = class;

  { TLayer Base class }

  TLayer = class
  private
    Params: TVariableArr;
  public
    function Eval(X: TVariable): TVariable; virtual; abstract;
    function GetParams: TVariableArr;
    procedure Cleanup;
  end;

  { TBatchNormLayer }

  TBatchNormLayer = class(TLayer)
  private
    FGamma, FBeta: TVariable;
  public
    constructor Create;
    function Eval(X: TVariable): TVariable; override;
    property Gamma: TVariable read FGamma write FGamma;
    property Beta: TVariable read FBeta write FBeta;
  end;

  { TConv2dLayer }

  TConv2dLayer = class(TLayer)
    constructor Create(InChannels, OutChannels, KernelSize: longint;
      Strides: longint = 1; Padding: longint = 0);
    function Eval(X: TVariable): TVariable; override;
  end;

  { TDenseLayer, or fully-connected layer }

  TDenseLayer = class(TLayer)
  public
    constructor Create(InSize, OutSize: longint);
    function Eval(X: TVariable): TVariable; override;
  end;

  { TDropoutLayer }

  TDropoutLayer = class(TLayer)
  private
    FDropoutRate: float;
    FUseDropout:  boolean;
    function GetUseDropout: boolean;
  public
    constructor Create(ADropoutRate: float);
    function Eval(X: TVariable): TVariable; override;
    property DropoutRate: float read FDropoutRate write FDropoutRate;
    property UseDropout: boolean read GetUseDropout write FUseDropout;
  end;

  { TFlattenLayer }

  TFlattenLayer = class(TLayer)
  public
    function Eval(X: TVariable): TVariable; override;
  end;

  { TLeakyReLULayer }

  TLeakyReLULayer = class(TLayer)
  private
    FAlpha: float;
  public
    constructor Create(AAlpha: float);
    function Eval(X: TVariable): TVariable; override;
    property Alpha: float read FAlpha write FAlpha;
  end;

  { TReLULayer }

  TReLULayer = class(TLayer)
  public
    function Eval(X: TVariable): TVariable; override;
  end;

  { TSigmoidLayer }

  TSigmoidLayer = class(TLayer)
  public
    function Eval(X: TVariable): TVariable; override;
  end;

  { TSoftMaxLayer }

  TSoftMaxLayer = class(TLayer)
  private
    FAxis: longint;
  public
    constructor Create(AAxis: longint);
    function Eval(X: TVariable): TVariable; override;
    property Axis: longint read FAxis write FAxis;
  end;

  { TTanhLayer }

  TTanhLayer = class(TLayer)
  public
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
    procedure AddParam(param: TVariable);
    procedure Cleanup;
  end;

  TBatchingResult = record
    Xbatches, ybatches: TTensorArr;
    BatchCount: longint;
  end;

{ Loss functions }
function AccuracyScore(predicted, actual: TTensor): float;
function BinaryCrossEntropyLoss(ypred, ytrue: TVariable): TVariable;
function CrossEntropyLoss(ypred, ytrue: TVariable): TVariable;
function L2Regularization(Model: TModel; Lambda: float = 0.001): TVariable;

{ Utilities }
function CreateBatch(X: TTensor; BatchSize: integer): TTensorArr;
function CreateBatch(X, y: TTensor; BatchSize: integer): TBatchingResult;
function LoadModel(filename: string): TModel;
procedure SaveModel(Model: TModel; filename: string);


implementation

function BinaryCrossEntropyLoss(ypred, ytrue: TVariable): TVariable;
var
  m: longint;
begin
  Assert(ypred.Size = ytrue.Size, MSG_ASSERTION_DIFFERENT_LENGTH);

  m      := ypred.Size;
  Result := -(1 / m) * Sum(ytrue * Log(ypred) + (1 - ytrue) * Log(1 - ypred));
end;

function CrossEntropyLoss(ypred, ytrue: TVariable): TVariable;
begin
  Assert(ypred.Size = ytrue.Size, MSG_ASSERTION_DIFFERENT_LENGTH);
  Result := -Mean(ytrue * Log(ypred));
end;

function L2Regularization(Model: TModel; Lambda: float): TVariable;
var
  param: TVariable;
begin
  Result := 0;
  for param in Model.Params do
    if not param.Name.StartsWith('Bias') then
      Result := Result + Sum(param * param);
  Result     := Lambda * Result;
end;


function CreateBatch(X: TTensor; BatchSize: integer): TTensorArr;
var
  i, OutSize: longint;
begin
  OutSize := ceil(X.Shape[0] / BatchSize);
  SetLength(Result, OutSize);
  for i := 0 to OutSize - 1 do
    Result[i] := GetRowRange(X, i * BatchSize,
      Math.min(BatchSize, X.Shape[0] - i * BatchSize));

end;

function CreateBatch(X, y: TTensor; BatchSize: integer): TBatchingResult;
var
  i, OutSize: longint;
begin
  Assert(X.Shape[0] = y.Shape[0], 'X and y have different height');
  OutSize := ceil(X.Shape[0] / BatchSize);

  SetLength(Result.Xbatches, OutSize);
  SetLength(Result.ybatches, OutSize);
  Result.BatchCount := OutSize;

  for i := 0 to OutSize - 1 do
  begin
    Result.Xbatches[i] := GetRowRange(X, i * BatchSize,
      Math.min(BatchSize, X.Shape[0] - i * BatchSize));
    Result.ybatches[i] := GetRowRange(y, i * BatchSize,
      Math.min(BatchSize, y.Shape[0] - i * BatchSize));
  end;

end;

function JSONArrayToFloatVector(arr: TJSONArray): TFloatVector;
var
  i: longint;
begin
  SetLength(Result, arr.Count);
  for i := 0 to arr.Count - 1 do
    Result[i] := arr[i].AsFloat;
end;

function FloatVectorToJSONArray(arr: array of NFloat): TJSONArray;
var
  i: longint;
begin
  Result := TJSONArray.Create;
  for i := 0 to high(arr) do
    Result.Add(arr[i]);
end;

function IntVectorToJSONArray(arr: array of longint): TJSONArray;
var
  i: longint;
begin
  Result := TJSONArray.Create;
  for i := 0 to high(arr) do
    Result.Add(arr[i]);
end;

function LoadModel(filename: string): TModel;
var
  JData: TJSONData;
  o: TJSONEnum;
  LayerName: string;
  layer: TLayer;
  sl: TStringList;
  DenseIn, DenseOut: longint;
begin
  Result := TModel.Create;

  sl := TStringList.Create;
  sl.LoadFromFile(filename);

  JData := GetJSON(sl.Text);
  for o in TJSONArray(JData) do
  begin
    LayerName := o.Value.FindPath('layer_name').AsString;

    case LayerName of
      'Dense':
      begin
        DenseIn  := TJSONArray(o.Value.FindPath('layer_data.weight_shape')).Items[0].AsInteger;
        DenseOut := TJSONArray(o.Value.FindPath('layer_data.weight_shape')).Items[1].AsInteger;
        layer    := TDenseLayer.Create(DenseIn, DenseOut);

        layer.Params[0] :=
          CreateTensor([DenseIn, DenseOut], JSONArrayToFloatVector(
          TJSONArray(o.Value.FindPath('layer_data.weight_val')))).ToVariable(True);
        layer.Params[1] :=
          CreateTensor(layer.Params[1].Shape, JSONArrayToFloatVector(
          TJSONArray(o.Value.FindPath('layer_data.bias_val')))).ToVariable(True);
        Result.AddLayer(layer);
      end;
      'Dropout':
      begin
        layer := TDropoutLayer.Create(
          o.Value.FindPath('layer_data.DropoutRate').AsFloat);
        Result.AddLayer(layer);
      end;
      'LeakyReLU':
      begin
        layer := TLeakyReLULayer.Create(
          o.Value.FindPath('layer_data.leakiness').AsFloat);
        Result.AddLayer(layer);
      end;
      'ReLU':
      begin
        layer := TReLULayer.Create;
        Result.AddLayer(layer);
      end;
      'SoftMax':
      begin
        layer := TSoftMaxLayer.Create(
          o.Value.FindPath('layer_data.axis').AsInteger);
        Result.AddLayer(layer);
      end;
    end;
  end;

  sl.Free;
end;

procedure SaveModel(Model: TModel; filename: string);
var
  layer: TLayer;
  o, LayerData: TJSONObject;
  LayersJSONArr: TJSONArray;
  a: array[0..1] of integer;
  sl: TStringList;
begin
  LayersJSONArr := TJSONArray.Create;

  for layer in Model.LayerList do
  begin
    if layer is TDenseLayer then
    begin
      LayerData := TJSONObject.Create(
        ['weight_val', FloatVectorToJSONArray(layer.Params[0].Data.val),
        'weight_shape', IntVectorToJSONArray(layer.Params[0].Data.Shape),
        'bias_val', FloatVectorToJSONArray(layer.Params[1].Data.Val),
        'bias_shape', IntVectorToJSONArray(layer.Params[1].Data.Shape)]);
      LayersJSONArr.Add(TJSONObject.Create(['layer_name', 'Dense',
        'layer_data', LayerData]));
    end;

    if layer is TDropoutLayer then
    begin
      LayerData := TJSONObject.Create(['DropoutRate', TDropoutLayer(layer).DropoutRate]);
      LayersJSONArr.Add(TJSONObject.Create(['layer_name', 'Dropout',
        'layer_data', LayerData]));
    end;

    if layer is TLeakyReLULayer then
    begin
      LayerData := TJSONObject.Create(['leakiness', TLeakyReLULayer(layer).Alpha]);
      LayersJSONArr.Add(TJSONObject.Create(['layer_name', 'LeakyReLU',
        'layer_data', LayerData]));
    end;

    if layer is TReLULayer then
      LayersJSONArr.Add(TJSONObject.Create(['layer_name', 'ReLU']));

    if layer is TSoftMaxLayer then
    begin
      LayerData := TJSONObject.Create(['axis',
        TSoftMaxLayer(layer).Axis]);
      LayersJSONArr.Add(TJSONObject.Create(['layer_name', 'SoftMax',
        'layer_data', LayerData]));
    end;
  end;

  sl      := TStringList.Create;
  sl.Text := LayersJSONArr.AsJSON;
  sl.SaveToFile(filename);

  sl.Free;
  LayersJSONArr.Free;
end;

function AccuracyScore(predicted, actual: TTensor): float;
var
  i: integer;
  tot: float;
begin
  tot := 0;
  for i := 0 to predicted.Size - 1 do
    { check if the sample is correctly classified (i.e., predicted = actual) }
    if predicted.GetAt(i) = actual.GetAt(i) then
      tot := tot + 1;
  Result  := tot / predicted.Size;
end;

{ TFlattenLayer }

function TFlattenLayer.Eval(X: TVariable): TVariable;
var
  i, sz: longint;
begin
  sz := 1;
  for i := 1 to X.NDims - 1 do
    sz   := sz * X.Shape[i];
  Result := Reshape(X, [X.Shape[0], sz]);
end;

{ TConv2dLayer }

constructor TConv2dLayer.Create(InChannels, OutChannels, KernelSize: longint;
  Strides: longint; Padding: longint);
var
  W, b: TVariable;
begin
  inherited Create;
  { Xavier weight initialization }
  W := RandomTensorNormal([OutChannels, InChannels, KernelSize, KernelSize]) *
    ((2 / (InChannels * KernelSize * KernelSize)) ** 0.5);
  b := CreateTensor([1, OutChannels, 1, 1], 0);

  b.Name := 'Bias' + IntToStr(b.ID);
  SetRequiresGrad([W, b], True);

  SetLength(self.Params, 2);
  self.Params[0] := W;
  self.Params[1] := b;
end;

function TConv2dLayer.Eval(X: TVariable): TVariable;
begin
  //PrintTensor(Conv2D(self.Params[0], self.Params[1], 0, 0, 1, 1));
  Result := Conv2D(X, self.Params[0], 0, 0, 1, 1) + self.Params[1];
end;

{ TBatchNormLayer }

constructor TBatchNormLayer.Create;
begin
  self.Beta  := 0;
  self.Gamma := 1;

  self.Beta.Data.ReshapeInplace([1, 1]);
  self.Gamma.Data.ReshapeInplace([1, 1]);

  self.Beta.RequiresGrad  := True;
  self.Gamma.RequiresGrad := True;
end;

function TBatchNormLayer.Eval(X: TVariable): TVariable;
var
  muB, varB: TVariable;
begin
  muB    := Mean(X, 0);
  varB   := Sum(Sqr(X - muB), 0) / X.Shape[0];
  Result := self.Gamma * ((X - muB) / Sqrt(varB + 1e-8)) + self.Beta;
end;

{ TTanhLayer }

function TTanhLayer.Eval(X: TVariable): TVariable;
begin
  Result := Tanh(X);
end;

{ TSigmoidLayer }

function TSigmoidLayer.Eval(X: TVariable): TVariable;
begin
  Result := 0.5 * (Tanh(X / 2) + 1);
end;

{ TLeakyReLULayer }

constructor TLeakyReLULayer.Create(AAlpha: float);
begin
  self.Alpha := AAlpha;
end;

function TLeakyReLULayer.Eval(X: TVariable): TVariable;
begin
  Result := LeakyReLU(X, self.FAlpha);
end;

{ TReLULayer }

function TReLULayer.Eval(X: TVariable): TVariable;
begin
  Result := ReLU(X);
end;

{ TDropoutLayer }

function TDropoutLayer.GetUseDropout: boolean;
begin
  if GLOBAL_SKIP_GRAD then
    exit(False)
  else
    Result := self.FUseDropout;
end;

constructor TDropoutLayer.Create(ADropoutRate: float);
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

constructor TDenseLayer.Create(InSize, OutSize: longint);
var
  W, b: TVariable;
begin
  inherited Create;

  { Xavier weight initialization }
  W      := TVariable.Create(RandomTensorNormal([InSize, OutSize]) *
    ((2 / (InSize + OutSize)) ** 0.5));
  b      := TVariable.Create(CreateTensor([1, OutSize], 0));
  b.Name := 'Bias' + IntToStr(b.ID);
  SetRequiresGrad([W, b], True);

  SetLength(self.Params, 2);
  self.Params[0] := W;
  self.Params[1] := b;
end;

function TDenseLayer.Eval(X: TVariable): TVariable;
begin
  Result := X.Dot(self.Params[0]) + self.Params[1];
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
    self.AddParam(param);
end;

procedure TModel.AddParam(param: TVariable);
begin
  SetLength(self.Params, Length(self.Params) + 1);
  self.Params[Length(self.Params) - 1] := param;
end;

procedure TModel.Cleanup;
var
  l: TLayer;
begin
  Params := nil;
  for l in LayerList do
  begin
    l.Cleanup;
    l.Free;
  end;
  FreeAndNil(LayerList);
  FreeAndNil(self);
end;

{ TLayer }

function TLayer.GetParams: TVariableArr;
begin
  Result := self.Params;
end;

procedure TLayer.Cleanup;
begin
  Params := nil;
end;

end.
