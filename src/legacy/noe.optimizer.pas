{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit provides implementation for neural network optimization algorithms.
}

unit noe.optimizer;

{$mode objfpc}{$H+}

interface

uses
  Classes, noe, noe.math, noe.utils, SysUtils;

procedure DefaultOptimizerCallback(Loss: TVariable; iteration: longint;
  Params: array of TVariable);

type
  TOptimizerCallbackProc = procedure(Loss: TVariable; iteration: longint;
    Params: array of TVariable);

  { The base class for optimizer. All optimizers should extend this class. }

  { TBaseOptimizer }

  TBaseOptimizer = class
  private
    FCallback:     TOptimizerCallbackProc;
    FLearningRate: double;
    FIteration:    longint;
    FVerbose:      boolean;
  public
    constructor Create;
    procedure UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
    procedure Cleanup;
    property LearningRate: double read FLearningRate write FLearningRate;
    property Iteration: longint read FIteration write FIteration;
    property Verbose: boolean read FVerbose write FVerbose;
  end;

  { The implementation of stochastic gradient descent. It is the most basic
    optimizer among available ones. }

  TSGDOptimizer = class(TBaseOptimizer)
    constructor Create;
    procedure UpdateParams(Loss: TVariable; ModelParams: array of TVariable);

  end;

  { The implementation of stochastic gradient descent with momentum }

  TSGDMomentumOptimizer = class(TBaseOptimizer)
  private
    FGamma: double;
    V:      array of TTensor;
    VPopulated: boolean;
  public
    constructor Create;
    procedure UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
    property Gamma: double read FGamma write FGamma;
  end;

  { The implementation of adam optimizer. It was proposed by Kingma & Ba (2014).
    Please check the paper, "Adam: A Method for Stochastic Optimization", here:
    https://arxiv.org/abs/1412.6980. }

  TAdamOptimizer = class(TBaseOptimizer)
  private
    M: array of TTensor;
    V: array of TTensor;
    MVPopulated: boolean;
  public
    Epsilon: double;
    Beta1:   double;
    Beta2:   double;
    constructor Create;
    procedure UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
  end;

  { TRMSPropOptimizer }

  TRMSPropOptimizer = class(TBaseOptimizer)
  private
    V: array of TTensor;
    VPopulated: boolean;
  public
    Epsilon: double;
    Gamma:   double;
    constructor Create;
    procedure UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
  end;

implementation

procedure DefaultOptimizerCallback(Loss: TVariable; iteration: longint;
  Params: array of TVariable);
begin
  NoeLog('Debug', 'Epoch ' + IntToStr(iteration) + ': loss = ' +
    FloatToStrF(Loss.Data.GetAt(0), ffFixed, 2, 5));
end;

{ TRMSPropOptimizer }

constructor TRMSPropOptimizer.Create;
begin
  inherited;
  self.LearningRate := 0.001;
  self.Epsilon    := 10E-8;
  self.Gamma      := 0.99;
  self.VPopulated := False;
end;

procedure TRMSPropOptimizer.UpdateParams(Loss: TVariable;
  ModelParams: array of TVariable);
var
  i: longint;
begin
  inherited;
  if not self.VPopulated then
  begin
    SetLength(self.V, Length(ModelParams));
    for i := 0 to Length(ModelParams) - 1 do
    begin
      self.V[i] := Zeros(ModelParams[i].Data.Shape);
    end;
    self.VPopulated := True;
  end;

  for i := 0 to Length(ModelParams) - 1 do
  begin
    self.V[i] := self.Gamma * self.V[i] + (1 - self.Gamma) * (ModelParams[i].Grad ** 2);

    { Model parameter update }
    ModelParams[i].Data := ModelParams[i].Data - self.LearningRate *
      ModelParams[i].Grad / ((self.V[i]) ** 0.5 + self.Epsilon);
  end;
end;

{ TBaseOptimizer }

constructor TBaseOptimizer.Create;
begin
  Self.Verbose   := True;
  Self.FCallback := @DefaultOptimizerCallback;
end;

procedure TBaseOptimizer.UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
begin
  ZeroGradGraph(Loss);
  Loss.Backpropagate;

  if self.Verbose then
    self.FCallback(Loss, self.FIteration, ModelParams);

  Inc(FIteration);
end;

procedure TBaseOptimizer.Cleanup;
begin
  FreeAndNil(self);
end;

{ TSGDMomentumOptimizer }

constructor TSGDMomentumOptimizer.Create;
begin
  inherited;
  self.LearningRate := 0.01;
  self.VPopulated   := False;
end;

procedure TSGDMomentumOptimizer.UpdateParams(Loss: TVariable;
  ModelParams: array of TVariable);
var
  i: integer;
begin
  inherited;

  if not self.VPopulated then
  begin
    SetLength(self.V, Length(ModelParams));
    for i := 0 to Length(ModelParams) - 1 do
      self.V[i]     := Zeros(ModelParams[i].Data.Shape);
    self.VPopulated := True;
  end;

  for i := 0 to Length(ModelParams) - 1 do
  begin
    self.V[i] := self.Gamma * self.V[i] + self.LearningRate * ModelParams[i].Grad;
    ModelParams[i].Data := ModelParams[i].Data - self.V[i];
  end;

end;

{ TAdamOptimizer }

constructor TAdamOptimizer.Create;
begin
  inherited;

  self.FIteration := 1;
  self.LearningRate := 0.001;
  self.Epsilon := 10E-8;
  self.Beta1 := 0.9;
  self.Beta2 := 0.999;
  self.MVPopulated := False;
end;

procedure TAdamOptimizer.UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
var
  mHat, vHat: TTensor;
  i: longint;
begin
  inherited;

  { initialize elements in M and V once with zeros }
  if not self.MVPopulated then
  begin
    SetLength(self.M, Length(ModelParams));
    SetLength(self.V, Length(ModelParams));
    for i := 0 to Length(ModelParams) - 1 do
    begin
      self.M[i] := Zeros(ModelParams[i].Data.Shape);
      self.V[i] := Zeros(ModelParams[i].Data.Shape);
    end;
    self.MVPopulated := True;
  end;

  for i := 0 to Length(ModelParams) - 1 do
  begin
    { First and second moment estimate }
    self.M[i] := self.Beta1 * self.M[i] + (1 - Self.Beta1) * ModelParams[i].Grad;
    self.V[i] := self.Beta2 * self.V[i] + (1 - Self.Beta2) * (ModelParams[i].Grad ** 2);

    { Bias correction }
    mHat := self.M[i] / (1 - (self.Beta1 ** (self.Iteration)));
    vHat := self.V[i] / (1 - (self.Beta2 ** (self.Iteration)));

    { Model parameter update }
    ModelParams[i].Data := ModelParams[i].Data - self.LearningRate *
      mHat / ((vHat ** 0.5) + self.Epsilon);
  end;
end;

{ TSGDOptimizer }

constructor TSGDOptimizer.Create;
begin
  inherited;

  self.LearningRate := 0.01;
end;

procedure TSGDOptimizer.UpdateParams(Loss: TVariable; ModelParams: array of TVariable);
var
  param: TVariable;
begin
  inherited;

  for param in ModelParams do
  begin
    param.Data := param.Data - self.LearningRate * param.Grad;
  end;

end;

end.
