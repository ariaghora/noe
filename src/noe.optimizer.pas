unit noe.optimizer;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, noe.core, noe.autograd;

type

  { TBaseOptimizer }

  TBaseOptimizer = class
  private
    FLearningRate: double;
    FLoss:   TVariable;
  public
    constructor Create(ALearningRate: double);
    procedure UpdateParams(params: array of TVariable); overload; virtual; abstract;
    property LearningRate: double read FLearningRate write FLearningRate;
    property Loss: TVariable read FLoss write FLoss;
  end;

  { TGradientDescentOptimizer }

  TGradientDescentOptimizer = class(TBaseOptimizer)
    procedure UpdateParams(ModelParams: array of TVariable); override;
  end;

implementation

{ TBaseOptimizer }

constructor TBaseOptimizer.Create(ALearningRate: double);
begin
  self.LearningRate := ALearningRate;
end;

{ TGradientDescentOptimizer }

procedure TGradientDescentOptimizer.UpdateParams(ModelParams: array of TVariable);
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
