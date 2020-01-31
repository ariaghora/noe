{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit provides implementation for neural network optimization algorithms.
}

unit noe.optimizer;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, noe;

type

  { The base class for optimizer. All optimizers should extend this class. }
  TBaseOptimizer = class
  private
    FLearningRate: double;
  public
    constructor Create(ALearningRate: double);
    procedure UpdateParams(params: array of TVariable); overload; virtual; abstract;
    property LearningRate: double read FLearningRate write FLearningRate;
  end;

  { The implementation of stochastic gradient descent. It is the most basic
    optimizer among available ones. }
  TSGDOptimizer = class(TBaseOptimizer)
    procedure UpdateParams(ModelParams: array of TVariable); override;
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
    iteration: longint;
    constructor Create;
    procedure UpdateParams(ModelParams: array of TVariable); override;
  end;

implementation

{ TAdamOptimizer }

constructor TAdamOptimizer.Create;
begin
  self.iteration := 1;
  self.LearningRate := 0.001;
  self.Epsilon := 10E-8;
  self.Beta1 := 0.9;
  self.Beta2 := 0.999;
  self.MVPopulated := False;
end;

procedure TAdamOptimizer.UpdateParams(ModelParams: array of TVariable);
var
  mHat, vHat: TTensor;
  i: longint;
begin
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
    mHat := self.M[i] / (1 - (self.Beta1 ** (self.iteration)));
    vHat := self.V[i] / (1 - (self.Beta2 ** (self.iteration)));

    { Model parameter update }
    ModelParams[i].Data := ModelParams[i].Data - self.LearningRate *
      mHat / ((vHat ** 0.5) + self.Epsilon);

    ModelParams[i].ZeroGrad;
  end;
  Inc(self.iteration);
end;


{ TBaseOptimizer }

constructor TBaseOptimizer.Create(ALearningRate: double);
begin
  self.LearningRate := ALearningRate;
end;

{ TSGDOptimizer }

procedure TSGDOptimizer.UpdateParams(ModelParams: array of TVariable);
var
  param: TVariable;
begin
  for param in ModelParams do
  begin
    param.Data := param.Data - self.LearningRate * param.Grad;

    param.ZeroGrad;
  end;
end;

end.
