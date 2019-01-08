unit DTMLUtils;

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCommon, DTLinAlg;

function CategoricalCrossEntropy(ypred, y: TFloatVector): real;
function Sigmoid(x: real): real;
function SigmoidPrime(x: real): real;
function Relu(x: real): real;
function ReluPrime(x: real): real;

implementation

{
  A collection of loss function. Note that the loss is performed row-wise.
  At least for now.
}
function CategoricalCrossEntropy(ypred, y: TFloatVector): real;
begin
  //Result := -sum(Multiply(y, ElementWise(@Log, ypred)));
end;


{
  A collection of nonlinear activation functions
}
function Sigmoid(x: real): real;
begin
  Result := 1 / (1 + exp(-x));
end;

function SigmoidPrime(x: real): real;
begin
  Result := Sigmoid(x) * (1 - Sigmoid(x));
end;

function Relu(x: real): real;
begin
  Result := x * integer(x > 0);
end;

function ReluPrime(x: real): real;
begin
  Result := 1 * integer(x > 0);
end;


end.

