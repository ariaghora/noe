unit noe.optimizer;

{$mode objfpc}{$H+}

interface

uses
  fgl, multiarray, numerik, noe;

type
  TOptimizer = class
  private
    FModelParams: TTensorList;
  public
    LearningRate: single;
    constructor Create(ModelParams: TTensorList); virtual;
    procedure Step; virtual; abstract;
  end;

  TOptSGD = class(TOptimizer)
  public
    constructor Create(ModelParams: TTensorList; ALearningRate: single); overload;
    procedure Step; override;
  end;

  TOptRMSPROP = class(TOptimizer)
  private
    V: array of TMultiArray;
  public
    Epsilon: single;
    Gamma: single;
    constructor Create(ModelParams: TTensorList); override;
    procedure Step; override;
  end;

implementation

constructor TOptimizer.Create(ModelParams: TTensorList);
begin
  self.FModelParams := ModelParams;
end;

constructor TOptSGD.Create(ModelParams: TTensorList; ALearningRate: single);
begin
  inherited Create(ModelParams);

  LearningRate := ALearningRate;
end;

procedure TOptSGD.Step;
var
  T: TTensor;
begin
  for T in FModelParams do
  begin
    T.Data := T.Data - T.Grad * LearningRate;
    T.ZeroGrad;
  end;
end;

constructor TOptRMSPROP.Create(ModelParams: TTensorList);
var
  i: integer;
begin
  inherited Create(ModelParams);
  LearningRate := 0.001;
  Epsilon := 10E-8;
  Gamma := 0.99;
  SetLength(V, ModelParams.Count);
  for i := 0 to ModelParams.Count - 1 do
    V[i] := Zeros(ModelParams[i].Shape);
end;

procedure TOptRMSPROP.Step;
var
  i: integer;
begin
  for i := 0 to FModelParams.Count - 1 do
  begin
    V[i] := Gamma * V[i] + LearningRate * FModelParams[i].Grad;
    FModelParams[i].Data := FModelParams[i].Data - self.V[i];
    FModelParams[i].ZeroGrad;
  end;
end;

end.

