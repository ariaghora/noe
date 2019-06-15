unit DTMLUtils;

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCommon, DTLinAlg;

function CategoricalCrossEntropy(ypred, y: TFloatVector): Double; overload;
function CategoricalCrossEntropy(ypred, y: TFloatMatrix): Double; overload;
function Sigmoid(x: Double): Double;
function SigmoidPrime(x: Double): Double;
function Relu(x: Double): Double;
function ReluPrime(x: Double): Double;

implementation

{
  A collection of loss function. Note that the loss is performed row-wise.
  At least for now.
}
function CategoricalCrossEntropy(ypred, y: TFloatVector): Double;
var
  i, m: integer;
begin
  m := Length(ypred);
  Result := 0;
  for i := 0 to m - 1 do
    Result := Result + (y[i] * ln(ypred[i]));
  //Result := -sum(Multiply(y, ElementWise(@DTCommon.Log, ypred)));
  Result := Result;
end;

function CategoricalCrossEntropy(ypred, y: TFloatMatrix): Double;
var
  i, m: integer;
begin
  Result := 0;
  m := Shape(ypred)[0];
  for i := 0 to m - 1 do
    Result := Result + CategoricalCrossEntropy(ypred[i], y[i]);
  //Result;
  Result := -Result;
end;

{
  A collection of nonlinear activation functions
}
function Sigmoid(x: Double): Double;
begin
  Result := 1 / (1 + exp(-x));
end;

function SigmoidPrime(x: Double): Double;
begin
  Result := Sigmoid(x) * (1 - Sigmoid(x));
end;

function Relu(x: Double): Double;
begin
  Result := x * integer(x > 0);
end;

function ReluPrime(x: Double): Double;
begin
  Result := 1 * integer(x > 0);
end;


end.

